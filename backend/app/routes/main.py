from flask import Blueprint, jsonify

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return jsonify({"message": "Welcome to the Flask Backend!"})

@main.route('/api/test')
def test_api():
    return jsonify({"message": "Hello from Flask!", "status": "success"})
