def cost (tokens, model):
    """
    Calculate the cost (in USD) of generating a given number of tokens using a specified language model.

    Args:
        tokens (int): The number of tokens to calculate the cost for.
        model (str): The language model to use for cost calculation (e.g., 'gpt-3.5-turbo', 'gpt-4').

    Returns:
        float: The cost in USD for generating the specified number of tokens with the chosen model.
    """
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