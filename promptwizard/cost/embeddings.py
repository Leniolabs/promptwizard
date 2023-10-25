def cost (tokens, model):
    """
    Calculate the cost (in USD) of generating a given number of tokens using a specified language model.

    Args:
        tokens (int): The number of tokens to calculate the cost for.
        model (str): The language model to use for cost calculation (e.g., 'text-embedding-ada-002'.

    Returns:
        float: The cost in USD for generating the specified number of tokens with the chosen model.
    """
    if tokens == 0:
        cost = 0

    if model == 'text-embedding-ada-002':
        cost = (0.0001/1000)*tokens

    if model == 'text-similarity-ada-001':
        cost = (0.004/1000)*tokens

    if model == 'text-search-ada-query-001':
        cost = (0.004/1000)*tokens

    if model == 'code-search-ada-code-001':
        cost = (0.004/1000)*tokens

    if model == 'code-search-ada-text-001':
        cost = (0.004/1000)*tokens

    if model == 'text-similarity-babbage-001' or model == 'text-search-babbage-doc-001' or model == 'text-search-babbage-query-001' or model == 'code-search-babbage-code-001' or model == 'code-search-babbage-text-001':
        cost = (0.005/1000)*tokens

    if model == 'text-similarity-curie-001' or model == 'text-search-curie-doc-001' or model == 'text-search-curie-query-001':
        cost = (0.02/1000)*tokens

    if model == 'text-similarity-davinci-001' or model == 'text-search-davinci-doc-001' or model == 'text-search-davinci-query-001':
        cost = (0.2/1000)*tokens

    return cost