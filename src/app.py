from flask import Flask, request, jsonify
import requests
import re
from helpers import get_predictions, deabstract
from barplot import plot_sdg_barplot_with_images

app = Flask(__name__)

@app.route('/classify/', methods=['POST'])
def classify():
    # use the param 'text' to get the text to classify
    text = request.json.get('text', None)
    if not text:
        return jsonify({"msg": "Missing 'text' in request data"}), 400

    try:
        result = get_predictions(text)
        return jsonify({"text": text, "predictions": result})
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
        
        # Extract fields for classification
        title = data.get('title', '')
        abstract = deabstract(data.get('abstract_inverted_index'))
        
        # Extract topics, keywords, and concepts
        topics = []
        if data.get('primary_topic'):
            for key in ['subfield', 'field', 'domain']:
                if data['primary_topic'].get(key) and data['primary_topic'][key].get('display_name'):
                    topics.append(data['primary_topic'][key]['display_name'])
        
        keywords = [kw.get('display_name') for kw in data.get('keywords', []) if kw.get('display_name')]
        concepts = [c.get('display_name') for c in data.get('concepts', []) if c.get('display_name')]
        mesh_terms = [m.get('descriptor') for m in data.get('mesh', []) if m.get('descriptor')]

        # Combine all text fields
        text_parts = [title, abstract] + topics + keywords + concepts + mesh_terms
        text_to_classify = ". ".join(filter(None, text_parts))

        result = get_predictions(text_to_classify)
        return jsonify({"text": text_to_classify, "predictions": result})
    
    except requests.exceptions.HTTPError as e:
        return jsonify({"msg": f"Failed to fetch data from OpenAlex: {str(e)}"}), e.response.status_code
    except Exception as e:
        return jsonify({"msg": str(e)}), 500
    

@app.route('/plot-predictions/', methods=['POST'])
def plot_predictions():
    """
    Accepts a DOI, gets SDG predictions, and returns a bar plot image.
    """
    doi = request.json.get('doi', None)
    if not doi:
        return jsonify({"msg": "Missing 'doi' in request data"}), 400

    try:
        # Sanitize DOI
        doi = re.sub(r'^https?://doi.org/', '', doi)
        
        # Fetch from OpenAlex
        url = f"https://api.openalex.org/works/doi:{doi}"
        r = requests.get(url)
        r.raise_for_status()
        
        data = r.json()
        
        # Extract fields for classification
        title = data.get('title', '')
        abstract = deabstract(data.get('abstract_inverted_index'))
        
        # Extract topics, keywords, and concepts
        topics = []
        if data.get('primary_topic'):
            for key in ['subfield', 'field', 'domain']:
                if data['primary_topic'].get(key) and data['primary_topic'][key].get('display_name'):
                    topics.append(data['primary_topic'][key]['display_name'])
        
        keywords = [kw.get('display_name') for kw in data.get('keywords', []) if kw.get('display_name')]
        concepts = [c.get('display_name') for c in data.get('concepts', []) if c.get('display_name')]
        mesh_terms = [m.get('descriptor') for m in data.get('mesh', []) if m.get('descriptor')]

        # Combine all text fields
        text_parts = [title, abstract] + topics + keywords + concepts + mesh_terms
        text_to_classify = ". ".join(filter(None, text_parts))
        
        predictions = get_predictions(text_to_classify)
        
        # Generate plot
        plot_data = plot_sdg_barplot_with_images(predictions)
        
        # Return image in an HTML response
        return f"<img src='data:image/png;base64,{plot_data}'/>"

    except requests.exceptions.HTTPError as e:
        return jsonify({"msg": f"Failed to fetch data from OpenAlex: {str(e)}"}), e.response.status_code
    except Exception as e:
        return jsonify({"msg": str(e)}), 500


if __name__ == "__main__":
    app.run()
