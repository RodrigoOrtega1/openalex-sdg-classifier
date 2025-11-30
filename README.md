### OpenAlex SDG Classifier

This API uses Aurora SDG queries and an Aurora machine learning model to map UN sustainable development goals to works.

Aurora SDG queries: https://aurora-network-global.github.io/sdg-queries/

Model source: https://zenodo.org/record/7304547

---

### Modificado para obtener una predicción a partir del DOI y que corra localmente  
Por Rodrigo Aldair Ortega Venegas

## Como usar
1. Descargar modelo de fuente, pesa 2GBs: https://zenodo.org/record/7304547

2. Instalar ambiente de conda
```
$ conda env create --file environment.yml
```
3. Activar el ambiente de conda
```
$ conda activate sdgclassifier
```
4. Descargar assets de nltk
```
$ python assets/download_nltk_data.py
```
Si se usa windows:
```
$ python .\assets\download_nltk_data.py
```
5. Crear un archivo .env con la ruta hacia el modelo  
Ejemplo de archivo .env:
```
MODEL_PATH="./model/SDG-BERT-v1.1_mbert_multilabel_model_based_on_aurora_sdg_queries_v5.h5"
```
6. Correr aplicación
```
$ python src/app.py
```
Si se usa windows:
```
$ python .\src\app.py
```
7. Usar Postman o curl para enviar peticiones al url /classify/ , /fetch-and-classify/ o /plot-predictions/  
Ejemplo:
```
curl -X POST \
  http://127.0.0.1:5000/classify/ \
  -H 'Content-Type: application/json' \
  -d '{"text": "Investment in clean and renewable energy sources is crucial for combating climate change and ensuring a sustainable future for all."}'
```
```
curl -X POST \
  http://127.0.0.1:5000/fetch-and-classify/ \
  -H 'Content-Type: application/json' \
  -d '{"doi": "10.1016/j.jneumeth.2015.01.020"}'
```
```
curl -X POST \
  http://127.0.0.1:5000/fetch-and-classify/ \
  -H 'Content-Type: application/json' \
  -d '{"doi": "10.1016/j.jneumeth.2015.01.020"}'
```
**NOTA:** Dependiendo si se tiene una GPU y el modelo, se podrá instalar una versión diferente de la biblioteca tensorflow
**NOTA:** Añadir paso para indicar como cargar las imagenes
