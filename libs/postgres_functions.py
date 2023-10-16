import os
from sqlalchemy import create_engine, MetaData, Table, text
import pandas as pd

def save_to_postgres(df, table_name, mode='replace', index=True):
    """
    Save a DataFrame to a PostgreSQL database.

    Args:
    - df (pd.DataFrame): The DataFrame to save.
    - table_name (str): The name of the table to save the DataFrame to.
    - mode (str, optional): The write mode, one of {'fail', 'replace', 'append'}. Defaults to 'replace'.
    - index (bool, optional): Whether to write the DataFrame's index to the SQL table. Defaults to True.
    
    Returns:
    - str: A success message.
    """
    # Validate the mode
    valid_modes = ['fail', 'replace', 'append']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}")

    # Get environment variables for the database connection
    DATABASE_USERNAME = os.environ.get("POSTGRES_USER")
    DATABASE_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    DATABASE_HOST = os.environ.get("POSTGRES_HOST")
    DATABASE_PORT = os.environ.get("POSTGRES_PORT")
    DATABASE_NAME = os.environ.get("POSTGRES_DB")

    # Create a connection string and engine
    connection_str = f"postgresql+psycopg2://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    engine = create_engine(connection_str)

    # Save the DataFrame to the database
    df.to_sql(table_name, engine, if_exists=mode, index=index)

    return f"DataFrame successfully saved to {table_name} in PostgreSQL using mode '{mode}'."


def read_from_postgres(table_name):
    """
    Read a table from a PostgreSQL database into a pandas DataFrame.

    Args:
    - table_name (str): The name of the table to read.

    Returns:
    - pd.DataFrame: The table contents as a DataFrame.
    """
    # Get environment variables for the database connection
    DATABASE_USERNAME = os.environ.get("POSTGRES_USER")
    DATABASE_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    DATABASE_HOST = os.environ.get("POSTGRES_HOST")
    DATABASE_PORT = os.environ.get("POSTGRES_PORT")
    DATABASE_NAME = os.environ.get("POSTGRES_DB")

    # Create a connection string and engine
    connection_str = f"postgresql+psycopg2://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    engine = create_engine(connection_str)

    # Read the table into a DataFrame
    df = pd.read_sql_table(table_name, engine)

    # Print success message and some info about the table
    print(f"Successfully loaded table '{table_name}' from PostgreSQL.")
    print(f"Number of records: {len(df)}")
    print(f"Columns: {', '.join(df.columns)}")

    return df

def drop_table_from_postgres(table_name):
    """
    Drop a table from a PostgreSQL database.

    Args:
    - table_name (str): The name of the table to drop.

    Returns:
    - str: A success or failure message.
    """
    # Get environment variables for the database connection
    DATABASE_USERNAME = os.environ.get("POSTGRES_USER")
    DATABASE_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    DATABASE_HOST = os.environ.get("POSTGRES_HOST")
    DATABASE_PORT = os.environ.get("POSTGRES_PORT")
    DATABASE_NAME = os.environ.get("POSTGRES_DB")

    # Create a connection string and engine
    connection_str = f"postgresql+psycopg2://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    engine = create_engine(connection_str)

    # Initialize metadata
    metadata = MetaData()

    # Create a Table object
    table_to_drop = Table(table_name, metadata)

    # Drop the table
    table_to_drop.drop(engine, checkfirst=True)

    return f"Table '{table_name}' has been dropped successfully."


def drop_database(db_name):
    """
    Drop a PostgreSQL database.

    Args:
    - db_name (str): The name of the database to drop.

    Returns:
    - str: A success or failure message.
    """
    # Get environment variables for the database connection
    DATABASE_USERNAME = os.environ.get("POSTGRES_USER")
    DATABASE_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    DATABASE_HOST = os.environ.get("POSTGRES_HOST")
    DATABASE_PORT = os.environ.get("POSTGRES_PORT")

    # Connection string to the default "postgres" database (which always exists)
    connection_str = f"postgresql+psycopg2://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/postgres"
    engine = create_engine(connection_str)

    # Drop the target database after disconnecting all sessions
    with engine.connect() as conn:
        # Terminate all sessions connected to the target database
        conn.execute(text(f"SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = '{db_name}' AND pid <> pg_backend_pid();"))
        # Drop the database
        conn.execute(text(f"DROP DATABASE {db_name};"))

    return f"Database '{db_name}' has been dropped successfully."

# Usage:
# message = drop_database('database_name')
# print(message)
