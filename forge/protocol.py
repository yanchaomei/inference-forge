"""Text-only OpenAI wire format around the fixed-length greedy runtime."""


class ProtocolError(ValueError):
    def __init__(self, message, status=400, param=None, code=None):
        super().__init__(message)
        self.status, self.param, self.code = status, param, code


def normalize_request(payload, *, chat, model, tokenizer):
    common = {'model','max_tokens','max_completion_tokens','stream','stream_options',
              'temperature','top_p','n','frequency_penalty','presence_penalty','stop','request_id'}
    allowed = common | ({'messages'} if chat else {'prompt','input_ids'})
    for name in payload:
        if name not in allowed:
            raise ProtocolError(f'Unsupported parameter: {name}', param=name, code='unsupported_parameter')
    if 'model' in payload and payload['model'] != model:
        raise ProtocolError(f'Model {payload["model"]!r} is not served; use {model!r}',
                            404, 'model', 'model_not_found')
    for name, expected in [('temperature',0),('top_p',1),('frequency_penalty',0),('presence_penalty',0)]:
        value = payload.get(name, expected)
        if type(value) not in (int,float) or value != expected:
            raise ProtocolError(f'{name} must be {expected} for fixed-length greedy generation', param=name)
    if type(payload.get('n',1)) is not int or payload.get('n',1) != 1:
        raise ProtocolError('Only n=1 is supported', param='n')
    if payload.get('stop') is not None:
        raise ProtocolError('stop is not supported; generation uses max_tokens', param='stop')
    if type(payload.get('stream',False)) is not bool:
        raise ProtocolError('stream must be boolean', param='stream')
    options = payload.get('stream_options')
    include_usage = False
    if options is not None:
        if not isinstance(options,dict) or set(options)-{'include_usage'} or type(options.get('include_usage',False)) is not bool:
            raise ProtocolError('stream_options supports only boolean include_usage', param='stream_options')
        if not payload.get('stream',False):
            raise ProtocolError('stream_options requires stream=true', param='stream_options')
        include_usage = options.get('include_usage',False)
    normalized = {k:payload[k] for k in ('prompt','input_ids','max_tokens','stream','request_id') if k in payload}
    if 'max_completion_tokens' in payload:
        if not chat or 'max_tokens' in payload:
            raise ProtocolError('Use max_completion_tokens only for chat and without max_tokens', param='max_completion_tokens')
        normalized['max_tokens'] = payload['max_completion_tokens']
    if chat:
        messages = payload.get('messages')
        if not isinstance(messages,list) or not messages:
            raise ProtocolError('messages must be a nonempty list', param='messages')
        for message in messages:
            if (not isinstance(message,dict) or set(message)-{'role','content'}
                    or message.get('role') not in ('system','user','assistant')
                    or not isinstance(message.get('content'),str)):
                raise ProtocolError('Each message needs a system/user/assistant role and text content', param='messages')
        try:
            normalized['input_ids'] = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        except (ValueError,TypeError) as exc:
            raise ProtocolError(f'Cannot apply this model chat template: {exc}', param='messages') from exc
    return normalized, include_usage


def envelope(request_id, model, created, *, chat, streaming=False):
    return {'id':('chatcmpl-' if chat else 'cmpl-') + request_id, 'created':created, 'model':model,
            'object':'chat.completion.chunk' if chat and streaming else ('chat.completion' if chat else 'text_completion')}


def response(message, model, created, *, chat, streaming=False):
    result = {**message, **envelope(message['request_id'],model,created,chat=chat,streaming=streaming)}
    finish = message.get('finish_reason')
    if chat:
        value = {} if streaming and finish else {'content':message.get('text','')}
        if not streaming:
            value['role'] = 'assistant'
        choice = {'index':0, 'delta' if streaming else 'message':value, 'finish_reason':finish}
    else:
        choice = {'index':0, 'text':'' if streaming and finish else message.get('text',''),
                  'logprobs':None, 'finish_reason':finish}
    result['choices'] = [choice]
    if streaming:
        result.pop('usage',None)
    return result


def role_chunk(request_id, model, created):
    return {**envelope(request_id,model,created,chat=True,streaming=True),
            'choices':[{'index':0,'delta':{'role':'assistant','content':''},'finish_reason':None}]}


def usage_chunk(message, model, created, *, chat):
    return {**envelope(message['request_id'],model,created,chat=chat,streaming=True),
            'choices':[], 'usage':message['usage']}


def error_response(error, status):
    return {'error':{'message':str(error), 'type':'server_error' if status>=500 else 'invalid_request_error',
                     'param':getattr(error,'param',None), 'code':getattr(error,'code',None)}}
