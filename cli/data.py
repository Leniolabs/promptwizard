import pandas as pd
import json

# Carga el JSON desde el archivo
with open("examples/classification/output.json", "r") as file:
    data = json.load(file)

test_data = data[5][1:]

resultados_df = pd.DataFrame()

for prompt in data[5][1:]:
    prompt_df = pd.DataFrame(prompt)
    # Concatena el DataFrame del prompt con el DataFrame principal
    resultados_df = pd.concat([resultados_df, prompt_df], ignore_index=True)

print(resultados_df)