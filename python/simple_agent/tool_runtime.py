"""Explicit operations for user code and the tool test harness; no agent decisions."""
import asyncio
import copy
import hashlib
from urllib.parse import urlsplit
from .errors import RunError
from .protocol import json_text, validate_download_url
from .files import StoredFile, json_wire_size
from .file_formats import identify_file, download_filename
from .limits import validate_media, validate_request, GEMINI_INLINE_BYTES, GEMINI_PDF_BYTES
from .providers import llm_request, token_request, validate_llm_config, llm_response, model_file_metadata, model_input_support
from contextvars import ContextVar
from .usage import usage_record
from .bridge import input_artifact_ids, artifact_allowed


class ToolRuntime:
    def __init__(self, context):
        self.ctx = context
        self._parent = ContextVar('tool_parent', default=None)
        self._tasks = {}
        self._workflow = ContextVar('workflow', default=None)
        self._node = ContextVar('workflow_node', default=None)
        self._node_permit = ContextVar('workflow_node_permit', default=None)
        self._node_lease = ContextVar('workflow_node_lease', default=None)
        self._llm_contexts = ContextVar('llm_contexts', default=None)

    @property
    def node_state(self):
        scope = self._workflow.get()
        if scope is None or self._node.get() is None:
            raise RunError('node_execution_forbidden', '노드 안에서만 실행 상태를 조회할 수 있습니다.')
        return copy.deepcopy(scope[1])

    @property
    def node_prompt(self):
        node = self._node.get()
        if node is None:
            raise RunError('node_execution_forbidden', '노드 안에서만 프롬프트를 조회할 수 있습니다.')
        return node.prompt_template

    async def run_child(self, name, payload):
        scope = self._workflow.get()
        lease = self._node_lease.get()
        if scope is None or lease is None or not lease['active']:
            raise RunError('node_execution_forbidden', '실행 중인 노드 안에서만 하위 작업을 호출할 수 있습니다.')
        task = asyncio.current_task()
        already = task in lease['tasks']
        lease['tasks'].add(task)
        try:
            return await scope[0].run_child(name, payload, self)
        finally:
            if not already: lease['tasks'].discard(task)

    async def run_children(self, calls, *, mode='parallel'):
        lease = self._node_lease.get()
        if lease is None or not lease['active']:
            raise RunError('node_execution_forbidden', '실행 중인 노드 안에서만 하위 작업을 호출할 수 있습니다.')
        task = asyncio.current_task()
        already = task in lease['tasks']
        lease['tasks'].add(task)
        try:
            return await self._run_children(calls, mode=mode)
        finally:
            if not already: lease['tasks'].discard(task)

    async def _run_children(self, calls, *, mode):
        if mode not in {'parallel', 'serial'}:
            raise RunError('invalid_child_execution', 'mode는 parallel 또는 serial이어야 합니다.')
        if not isinstance(calls, list) or any(not isinstance(c, dict) or set(c) != {'node', 'payload'} for c in calls):
            raise RunError('invalid_child_execution', '하위 작업은 node와 payload 목록으로 지정하세요.')
        # Validate the entire declaration before starting any child.
        parent = self._node.get()
        scope = self._workflow.get()
        if parent is None or scope is None:
            raise RunError('node_execution_forbidden', '노드 안에서만 하위 작업을 호출할 수 있습니다.')
        for call in calls:
            if call['node'] not in parent.child_nodes:
                raise RunError('child_node_not_allowed', '선언되지 않은 하위 작업입니다.')
            spec = scope[0].nodes[call['node']]
            self.ctx.registry.authorize(spec, self.ctx)
            self.ctx.registry.validate_arguments(spec, call['payload'], self.ctx)
        group_id = self.ctx.deps.new_id()
        await self.ctx.effect('child_group', {'mode': mode, 'calls': calls}, lambda: None,
                              call_id=group_id, parent_id=self.parent_id)
        async def call(item):
            token = self._parent.set(group_id)
            try:
                output = await self.run_child(item['node'], item['payload'])
                return {'node': item['node'], 'status': 'completed', 'output': output, 'error': None}
            except Exception as exc:
                error = self.ctx.error(exc)
                return {'node': item['node'], 'status': 'indeterminate' if error['code']=='indeterminate' else 'failed',
                        'output': None, 'error': error}
            finally:
                self._parent.reset(token)
        return await asyncio.gather(*(call(item) for item in calls)) if mode == 'parallel' else [await call(item) for item in calls]

    async def select_next_node(self, prompt, files, *, config):
        """One explicit routing generation; it only returns a transition for the runner."""
        scope, current = self._workflow.get(), self._node.get()
        if scope is None or current is None:
            raise RunError('node_execution_forbidden', '노드 안에서만 다음 노드를 선택할 수 있습니다.')
        if {'tools', 'tool_choice', 'parallel_tool_calls'}.intersection(config):
            raise RunError('invalid_llm_config', '노드 선택의 도구·호출 제약은 흐름 선언에서 지정합니다.')
        candidates = [scope[0].nodes[name] for name in current.next_nodes
                      if scope[0].nodes[name].permission in self.ctx.permissions]
        schemas = [node.model_schema(strict=True) for node in candidates]
        result = await self.callLLM(prompt, files, config={**config, 'tools': schemas,
                                       'tool_choice': 'required', 'parallel_tool_calls': False})
        call = result['tool_calls'][0]
        context = self._llm_contexts.get().get(current.kind)
        if context is not None:
            context.routes[current.name] = call
        return {'next_node': call['name'], 'payload': call['arguments']}

    async def add_llm_tool_result(self, call_id, output):
        """Attach an actual tool outcome to the active stage; never executes the tool."""
        node, contexts = self._node.get(), self._llm_contexts.get()
        lease = self._node_lease.get()
        if node is None or contexts is None or node.kind not in contexts or lease is None or not lease['active']:
            raise RunError('node_execution_forbidden', '문맥을 유지하는 노드 안에서 함수 결과를 추가하세요.')
        async with contexts[node.kind].lock:
            contexts[node.kind].add_tool_result(call_id, output)

    @property
    def parent_id(self):
        return self._parent.get()

    async def invoke(self, name, arguments):
        self.ctx.assert_open()
        task = asyncio.current_task()
        lease = self._node_lease.get()
        if lease is not None: lease['tasks'].add(task)
        self._tasks[task] = self._tasks.get(task, 0) + 1
        try:
            return await self._invoke(name, arguments)
        finally:
            self._tasks[task] -= 1
            if not self._tasks[task]:
                del self._tasks[task]
                if lease is not None: lease['tasks'].discard(task)

    async def close(self):
        self.ctx.closed = True
        tasks = [task for task in self._tasks if task is not asyncio.current_task() and not task.done()]
        for task in tasks:
            if not task.cancelling():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return bool(tasks)

    async def _invoke(self, name, arguments):
        ctx = self.ctx
        tid = ctx.deps.new_id()
        record = {'name': name, 'arguments': copy.deepcopy(arguments), 'callId': tid,
                  'traceId': tid, 'parentCallId': self.parent_id, 'authorized': False, 'status': 'started'}
        tool_index=len(ctx.run['toolCalls'])
        ctx.run['toolCalls'].append(record)
        await ctx.stage('tool_started', {'toolIndex': tool_index}, 'started', call_id=tid, parent_id=self.parent_id)
        token = self._parent.set(tid)
        try:
            spec = ctx.registry.get(name)
            from .workflow import NodeSpec
            lease = self._node_lease.get()
            if self._workflow.get() is not None and (lease is None or not lease['active']):
                raise RunError('node_execution_forbidden', '종료된 노드에서는 도구를 실행할 수 없습니다.')
            if isinstance(spec, NodeSpec):
                scope = self._workflow.get()
                if scope is None or self._node_permit.get() != name or self._node.get() is not spec or scope[0].nodes.get(name) is not spec:
                    raise RunError('node_execution_forbidden', '노드는 허용된 흐름 전이를 통해서만 실행할 수 있습니다.')
                self._node_permit.set(None)
                record['node'] = {'kind': spec.kind.value, 'childOnly': spec.child_only,
                                  'dataRefs': list(spec.data_refs), 'promptTemplate': spec.prompt_template,
                                  'systemPrompt': spec.system_prompt}
            elif self._workflow.get() is not None:
                flow, _ = self._workflow.get()
                node = self._node.get()
                if node is None or name not in flow.allowed_tools(node):
                    raise RunError('node_tool_not_allowed', '현재 단계와 노드에 허용되지 않은 도구입니다.', details={'node': node.name if node else None, 'tool': name})
            ctx.registry.authorize(spec, ctx)
            record.update(authorized=True, readOnly=spec.read_only)
            ctx.registry.validate_arguments(spec, arguments, ctx)
            if ctx.run['mode'] == 'preview':
                raise RunError('preview_execution_forbidden', '미리보기에서는 도구를 실행하지 않습니다.')
            # The declaration is replayed separately from its file/model operations.
            # Never skip nested evidence by replaying an outer aggregate tool result.
            await ctx.effect('tool_call', {'name': name, 'arguments': arguments}, lambda: None,
                             call_id=tid, parent_id=record['parentCallId'])
            if spec.external:
                async def external_call():
                    value=await ctx.registry.invoke(spec, arguments, ctx)
                    ctx.registry.validate_result(spec,value,ctx)
                    return value
                result = await ctx.effect('external_tool', {'name':name,'arguments':arguments},
                    external_call, parent_id=tid, changing=not spec.read_only)
            else:
                result = await ctx.registry.invoke(spec, arguments, ctx)
            ctx.registry.validate_result(spec, result, ctx)
            record.update(result=copy.deepcopy(result), status='completed')
            await ctx.stage('tool_result', {'toolIndex': tool_index}, call_id=tid, parent_id=record['parentCallId'])
            return result
        except (Exception, asyncio.CancelledError) as exc:
            record.update(error=ctx.error(exc), status='indeterminate' if getattr(exc, 'code', '') == 'indeterminate' else 'failed')
            await ctx.stage('tool_result', {'toolIndex': tool_index, 'error': record['error']}, record['status'], call_id=tid, parent_id=record['parentCallId'])
            raise
        finally:
            self._parent.reset(token)

    async def file(self, ref):
        """Explicit read/download. Input receipt never calls this function."""
        meta = await self.invoke('file', {'ref': ref})
        return meta, StoredFile(self.ctx.store, meta)

    async def file_metadata(self, ref):
        return await self.invoke('file_metadata', {'ref': ref})

    async def _file_metadata(self, ref):
        ctx = self.ctx
        async def inspect():
            if not ref.startswith('https://') or urlsplit(ref).hostname == 'fixtures.simple-agent.invalid':
                aid = urlsplit(ref).path.strip('/') if ref.startswith('https://') else ref
                try:
                    meta = await ctx.store.get_meta(aid)
                except Exception as exc:
                    return {'ref': ref, 'filename': None, 'filenameSource': None, 'mime': None,
                            'bytes': None, 'metadataSource': 'stored', 'status': 'unavailable', 'error': ctx.error(exc)}
                if not artifact_allowed(meta, ctx.run['id'], input_artifact_ids(ctx.run)) and meta['id'] not in ctx.restored_artifact_ids:
                    raise RunError('artifact_scope', '현재 입력에 지정된 파일만 조회할 수 있습니다.')
                return {'ref': ref, 'artifactId': meta['id'], 'filename': meta['filename'], 'filenameSource': 'stored',
                        'mime': meta['mime'], 'bytes': meta['bytes'], 'metadataSource': 'stored', 'status': 'available'}
            validate_download_url(ref)
            try: name = download_filename(ref)
            except RunError: name = None
            base = {'ref': ref, 'filename': name, 'filenameSource': 'url' if name else None,
                    'mime': None, 'bytes': None, 'metadataSource': 'head'}
            try:
                from .io_retry import transfer
                async def head(): return await ctx.transport.http('HEAD', ref)
                status, headers, _ = await transfer(head, safe=True, events=ctx.run.setdefault('transfers', []), operation='file_metadata')
                headers = {k.lower(): v for k, v in headers.items()}
                if not 200 <= status < 300:
                    return {**base, 'status': 'unavailable', 'error': {'httpStatus': status}}
                disposition = headers.get('content-disposition', '')
                if disposition:
                    from email.message import Message
                    header = Message()
                    header['Content-Disposition'] = disposition
                    if header.get_filename():
                        base.update(filename=download_filename(ref, disposition), filenameSource='content-disposition')
                size = headers.get('content-length', '')
                return {**base, 'mime': headers.get('content-type', '').split(';')[0] or None,
                        'bytes': int(size) if size.isascii() and size.isdigit() else None, 'status': 'available'}
            except Exception as exc:
                return {**base, 'status': 'unavailable', 'error': ctx.error(exc)}
        return await ctx.effect('file_metadata', {'ref': ref}, inspect, parent_id=self.parent_id)

    async def _file(self, ref):
        ctx = self.ctx
        async def resolve():
            if ref.startswith('https://'):
                fixture = urlsplit(ref).hostname == 'fixtures.simple-agent.invalid'
                validate_download_url(ref, fixture=fixture)
                meta = (await ctx.store.get_meta(urlsplit(ref).path.strip('/')) if fixture else
                        await ctx.transport.download_to_store(ref, ctx.store, ctx.run['id']))
            else:
                meta = await ctx.store.get_meta(ref)
            if not artifact_allowed(meta, ctx.run['id'], input_artifact_ids(ctx.run)) and meta['id'] not in ctx.restored_artifact_ids:
                raise RunError('artifact_scope', '현재 입력에 명시한 원본 또는 이 실행의 산출물만 사용할 수 있습니다.')
            return meta
        meta = await ctx.effect('file_resolve', {'ref': ref}, resolve, parent_id=self.parent_id)
        file = StoredFile(ctx.store, meta)
        # The original bytes are evidence; replay also checks their local stored hash.
        hasher = hashlib.sha256()
        async for chunk in file.chunks():
            hasher.update(chunk)
        if hasher.hexdigest() != meta['sha256']:
            raise RunError('replay_corrupt', '원본 파일의 해시가 다릅니다.')
        format_info = identify_file(await file.prefix(), meta['filename'], meta['mime'])
        meta = {**meta, 'format': format_info, 'modelInputs': model_input_support(format_info)}
        if meta['id'] not in ctx.run['artifactIds']:
            ctx.run['artifactIds'].append(meta['id'])
        await ctx.stage('file_received', {'ref': ref, **meta}, parent_id=self.parent_id)
        return meta

    async def callLLM(self, prompt, files, *, config):
        return await self.invoke('callLLM', {'prompt':prompt, 'files':files, 'config':config})

    async def save_data(self, scope, key, value):
        return await self.invoke('save_data', {'scope': scope, 'key': key, 'value': value})

    async def load_data(self, scope, key):
        return await self.invoke('load_data', {'scope': scope, 'key': key})

    async def embed_text(self, text, *, dimensions=None):
        args = {'text': text}
        if dimensions is not None:
            args['dimensions'] = dimensions
        return await self.invoke('embed_text', args)

    async def rag_put(self, scope, key, text, *, vector, space, metadata):
        return await self.invoke('rag_put', {'scope': scope, 'key': key, 'text': text,
                                             'vector': vector, 'space': space, 'metadata': metadata})

    async def rag_search(self, scope, *, query, vector, space, limit):
        return await self.invoke('rag_search', {'scope': scope, 'query': query, 'vector': vector, 'space': space, 'limit': limit})

    async def rag_chunk(self, *, mode, **arguments):
        return await self.invoke('rag_chunk', {'mode': mode, **arguments})

    async def rag_index(self, scope, document_id, *, source, chunks, space, vectors):
        return await self.invoke('rag_index', {'scope': scope, 'document_id': document_id,
            'source': source, 'chunks': chunks, 'space': space, 'vectors': vectors})

    async def rag_retrieve(self, scope, *, query, vector, space, child_limit, parent_limit):
        return await self.invoke('rag_retrieve', {'scope': scope, 'query': query, 'vector': vector,
            'space': space, 'child_limit': child_limit, 'parent_limit': parent_limit})

    async def rag_delete(self, scope, document_id):
        return await self.invoke('rag_delete', {'scope': scope, 'document_id': document_id})

    async def _embed_text(self, text, dimensions=None):
        from .rag import embedding_request, embedding_result
        request = embedding_request(text, dimensions)
        payload, call_id = await self.model(request, lambda value: embedding_result(value, dimensions))
        result = embedding_result(payload, dimensions)
        result['callId'] = call_id
        result['usage'] = copy.deepcopy(next(u for u in self.ctx.run['usage'] if u['callId'] == call_id))
        return result

    async def _call_llm(self, prompt, files, *, config):
        options = validate_llm_config(config)
        node, scope = self._node.get(), self._workflow.get()
        context = None
        if node is not None and scope is not None:
            from .workflow import NodeKind
            from .llm_context import LLMContext
            system = scope[0].stage_prompts.get(node.kind, node.system_prompt)
            if system is not None:
                if options.get('system_instruction') not in (None, system):
                    raise RunError('llm_context_system_changed', '선언한 시스템 프롬프트와 호출 설정이 다릅니다.', details={'node': node.name})
                options = {**options, 'system_instruction': system}
            if node.prompt_template:
                prompt = node.prompt_template + ('\n\n' + prompt if prompt else '')
            if node.kind == NodeKind.RETRIEVAL and options.get('history'):
                raise RunError('retrieval_history_not_allowed', '리트리벌은 호출마다 독립 입력을 사용합니다. 필요한 자료를 이번 prompt에 전달하세요.')
            if node.kind in {NodeKind.PLANNING, NodeKind.GENERATE}:
                context = self._llm_contexts.get().setdefault(node.kind, LLMContext())
        if context is None:
            return await self._perform_llm(prompt, files, options)
        async with context.lock:
            context.prepare(options)
            return await self._perform_llm(prompt, files, options, context)

    async def _perform_llm(self, prompt, files, options, context=None):
        ctx = self.ctx
        if ctx.run['mode']=='live':
            # Guard before even resolving files. Pure user functions remain available.
            if not ctx.config.live_api_enabled:
                raise RunError('live_api_disabled', '실제 모델 API 호출이 비활성화되어 있습니다.')
            ctx.transport.key(options['provider'])
        resolved=[]
        for ref in files:
            meta, file = await self.file(ref)
            meta = {**meta, **model_file_metadata(meta['format'], options['provider'])}
            validate_media(meta, options['provider'])
            resolved.append((meta,file,None))
        request=llm_request(prompt,resolved,options)
        if options['provider']=='gemini' and (any(m['kind']=='video' for m,_,_ in resolved) or json_wire_size(request['body']) > (GEMINI_PDF_BYTES if any(m['mime']=='application/pdf' for m,_,_ in resolved) else GEMINI_INLINE_BYTES)):
            uploaded=[]
            for meta,file,_ in resolved:
                uri,receipt=await ctx.effect('media_upload', {'artifactId':meta['id'],'sha256':meta['sha256']},
                    lambda m=meta,f=file: ctx.transport.upload_file(f,m,ctx.run),parent_id=self.parent_id,changing=True)
                uploaded.append((meta,file,uri))
                await ctx.stage('media_upload_completed', {'artifactId':meta['id'],'receipt':receipt},parent_id=self.parent_id)
            resolved=uploaded
            request=llm_request(prompt,resolved,options)
        current_input = request['body'].get('input')
        if context is not None:
            current_input = copy.deepcopy(current_input)
            request['body']['input'] = [*copy.deepcopy(context.items), *current_input]
            request['body']['include'] = ['reasoning.encrypted_content']
            # Returned call IDs must remain unique across this entire conversation.
            options = {**options, 'history': context.items}
        validate_request(request,json_wire_size(request['body']))
        if options['max_input_tokens'] is not None:
            count_request=token_request(request)
            count=await ctx.effect('token_count',count_request,lambda:ctx.transport.count_tokens(count_request,ctx.run),parent_id=self.parent_id)
            key='totalTokens' if options['provider']=='gemini' else 'input_tokens'
            actual=count.get(key)
            if type(actual) is not int or actual < 0:
                raise RunError('invalid_token_count','공식 계수 응답에 입력 토큰 수가 없습니다.',details={'response':count})
            if actual > options['max_input_tokens']:
                raise RunError('caller_input_token_limit','호출자가 config에 명시한 입력 토큰 수를 초과했습니다.',details={'actual':actual,'configured':options['max_input_tokens']})
        payload,call_id=await self.model(request, lambda value: llm_response(value,request['provider'],request['api'],options))
        text, calls = llm_response(payload,request['provider'],request['api'],options)
        if context is not None:
            context.complete(options, current_input, payload['output'])
        return {'text':text, 'tool_calls':calls, 'response':payload,
                'files':[{'ref':ref,'artifactId':meta['id'],'sha256':meta['sha256']} for ref,(meta,_,_) in zip(files,resolved,strict=True)],
                'model':options['model'],'provider':options['provider'],'callId':call_id,
                'usage':copy.deepcopy(next(u for u in ctx.run['usage'] if u['callId']==call_id))}

    async def model(self, request, validate):
        ctx = self.ctx
        ctx.assert_open()
        call_id = ctx.deps.new_id()
        record = {'index': len(ctx.run['modelCalls']), 'callId': call_id, 'parentCallId': self.parent_id,
                  'request': copy.deepcopy(request), 'role': request['role'], 'model': request['body']['model'], 'status': 'started', 'startedAt': ctx.now()}
        ctx.run['modelCalls'].append(record)
        payload = {}
        try:
            async def generate():
                value = await ctx.transport.model(request, ctx.run)
                try:
                    if not isinstance(value, dict):
                        raise ValueError('model response must be an object')
                    validate(value)
                except Exception as exc:
                    if isinstance(exc, RunError) and exc.code in {'model_incomplete', 'model_refused'}:
                        raise
                    raise RunError('indeterminate', '모델 성공 응답을 해석할 수 없어 결과를 확인하지 못했습니다.',
                                   details={'response': value}) from exc
                return value
            payload = await ctx.effect('model', request, generate,
                                       call_id=call_id, parent_id=self.parent_id)
            record.update(response=copy.deepcopy(payload), status='completed')
        except (Exception, asyncio.CancelledError) as exc:
            error = ctx.error(exc)
            record.update(error=error, status='indeterminate' if error['code']=='indeterminate' else 'failed')
            payload = record['error'].get('details', {}).get('response', {})
            raise
        finally:
            usage = usage_record(payload if isinstance(payload, dict) else {}, request['provider'], request['body']['model'],
                                 origin='replay' if ctx.replay else ctx.run['mode'], at=ctx.now())
            if ctx.replay:
                source_id = ctx.replay_ids.get(call_id)
                if source_id:
                    old = next((u for u in ctx.replay.get('usage', []) if u.get('callId')==source_id), None)
                    if old: usage = {**copy.deepcopy(old), 'origin':'replay', 'originalCallId':source_id}
            usage['callId'] = call_id
            ctx.run['usage'].append(copy.deepcopy(usage))
            await ctx.stage('model_response', {'callIndex': record['index'], 'usage':usage}, record['status'], call_id=call_id, parent_id=self.parent_id)
        return payload, call_id
