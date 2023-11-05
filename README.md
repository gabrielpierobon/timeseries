# Time Series Analysis Environment with Docker Compose

This repository orchestrates a multi-container Docker environment for time series analysis, which includes Jupyter for interactive notebooks, Apache Airflow for scheduling workflows, PostgreSQL for database services, Grafana for visualization, MLflow for ML lifecycle management, MinIO for S3-like storage, and a Flask application for web services.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Docker installed on your machine.
- Docker Compose installed on your machine.
- Basic knowledge of Docker and containerization.

## Services Setup

### Building the Containers

After cloning the repository to your local machine, navigate to the directory where the `docker-compose.yml` file is located and run the following command to build the images and start the containers:

```bash
docker-compose up --build
```

This command will:

- Build the Jupyter, Airflow, and FlaskApp images based on their respective Dockerfiles.
- Pull the PostgreSQL, Grafana, MLflow, and MinIO images from Docker Hub.
- Create and start the containers for all the services defined in `docker-compose.yml`.

### Managing Containers

To stop the containers without removing them, you can run:

```bash
docker-compose stop
```

To stop the services and remove the containers, networks, volumes, and images created by `up`, you can use:

```bash
docker-compose down
```

If you want to remove the volumes along with the containers, you can run:

```bash
docker-compose down --volumes
```

### Interacting with Services

Each service can be accessed through its respective port as defined in the `docker-compose.yml` file:

- **Jupyter Notebook**: Visit `http://localhost:8888`.
- **Apache Airflow**: Access the web UI at `http://localhost:8080`.
- **Grafana**: Open `http://localhost:3000`.
- **MLflow**: Navigate to `http://localhost:5000`.
- **MinIO**: Access the S3-like storage UI at `http://localhost:9001`.
- **FlaskApp**: The web service is available at `http://localhost:5001`.
- **PgAdmin**: For PostgreSQL management, go to `http://localhost:5050`.

- **Full Microservices Console**: Visit `http://localhost:5001`.

### Environment Variables

Some services like PostgreSQL, Airflow, and MLflow require environment variables. These are defined in the `docker-compose.yml`. You can also use a `.env` file to keep sensitive data like passwords and usernames:

Create a `.env` file in the root directory of the project and add your variables:

```env
# Example content of .env file
POSTGRES_USER=airflow
POSTGRES_PASSWORD=yourpassword
AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:yourpassword@postgres:5432/airflow
```

Remember to replace `yourpassword` with your actual password and update the `docker-compose.yml` to use the `.env` file variables where applicable.

## Project Structure

- `airflow-dags/`: Place your Airflow DAG files here.
- `app/`: Source code for the Flask application.
- `datasets/`: Directory for storing datasets used in analyses.
- `dags/`: Directory to keep example DAGs.
- `libs/`: Custom libraries or dependencies.
- `mlruns/`: MLflow tracking and artifacts directory.
- `config/`, `logs/`, `outputs/`, `plots/`: Configurations, logs, analysis outputs, and plots directories respectively.

## Contributing

Contributions to this project are welcome. Please ensure any pull request or issues you open are descriptive and provide all necessary information.

## Support

If you encounter any problems, please file an issue along with a detailed description.

## License

This project is licensed under the MIT License - see the LICENSE file in the repository for details.

## Acknowledgments

We gratefully acknowledge the contributors to the open-source projects that make this analysis environment possible.