from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

# Replace 'http://localhost:PORT' with the actual URLs you would access the services at.
SERVICE_URLS = {
    'jupyter': 'http://localhost:8888',
    'airflow': 'http://localhost:8080',
    'pgadmin': 'http://localhost:5050',
    'grafana': 'http://localhost:3000',
    'mlflow': 'http://localhost:5000',
    'minio': 'http://localhost:9091'
}

@app.route('/')
def index():
    # The index route could render a template with links to each service
    return render_template('index.html', services=SERVICE_URLS)

@app.route('/goto/<service_name>')
def goto_service(service_name):
    # This route redirects to the actual service based on the service name provided
    url = SERVICE_URLS.get(service_name)
    if url:
        return redirect(url)
    else:
        return "Service not found", 404

# Add other routes as needed to interact with the services
# ...

if __name__ == '__main__':
    # Flask will run on port 5001 internally within the Docker container
    app.run(debug=True, host='0.0.0.0', port=5001)
