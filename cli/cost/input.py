def cost (tokens, model):
    if tokens == 0:
        cost = 0
    if model == 'gpt-3.5-turbo':
        cost = (0.0015/1000) * tokens
    if model == 'gpt-3.5-turbo-16K':
        cost = (0.003/1000) * tokens
    if model == 'gpt-4':
        cost = (0.03/1000) * tokens
    if model == 'gpt-4-32K':
        cost = (0.06/1000) * tokens
    return cost