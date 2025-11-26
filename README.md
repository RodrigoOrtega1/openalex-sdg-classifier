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
5. Correr aplicación
```
$ python src/app.py
```
6. Usar Postman o curl para enviar peticiones al url /classify/ o /fetch-and-classify/  
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
