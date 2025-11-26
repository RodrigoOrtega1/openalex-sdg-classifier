from flask import Flask, request, jsonify
import requests
import re
from helpers import get_predictions, deabstract

app = Flask(__name__)

@app.route('/classify/', methods=['POST'])
def classify():
    # use the param 'text' to get the text to classify
    text = request.json.get('text', None)
    if not text:
        return jsonify({"msg": "Missing 'text' in request data"}), 400

    try:
        result = get_predictions(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"msg": str(e)}), 500


@app.route('/fetch-and-classify/', methods=['POST'])
def fetch_and_classify():
    # use the param 'doi' to get the doi to classify
    doi = request.json.get('doi', None)
    if not doi:
        return jsonify({"msg": "Missing 'doi' in request data"}), 400

    try:
        # Sanitize DOI
        doi = re.sub(r'^https?://doi.org/', '', doi)
        
        # Fetch from OpenAlex
        url = f"https://api.openalex.org/works/doi:{doi}"
        r = requests.get(url)
        r.raise_for_status() # Raises an exception for bad status codes
        
        data = r.json()
        title = data.get('title', '')
        abstract = deabstract(data.get('abstract_inverted_index'))
        
        text_to_classify = f"{title}. {abstract}"
        
        result = get_predictions(text_to_classify)
        return jsonify(result)
    except requests.exceptions.HTTPError as e:
        return jsonify({"msg": f"Failed to fetch data from OpenAlex: {str(e)}"}), e.response.status_code
    except Exception as e:
        return jsonify({"msg": str(e)}), 500


if __name__ == "__main__":
    app.run()
