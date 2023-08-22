def cost (tokens, model):
    if tokens == 0:
        cost = 0
    if model == 'gpt-3.5-turbo':
        cost = (0.002 /1000) * tokens
    if model == 'gpt-3.5-turbo-16K':
        cost = (0.004/1000) * tokens
    if model == 'gpt-4':
        cost = (0.06/1000) * tokens
    if model == 'gpt-4-32K':
        cost = (0.12/1000) * tokens
    return cost