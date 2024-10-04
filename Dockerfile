# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory in the container
WORKDIR /Portfolio-Optimizer

# Copy the requirements file and install dependencies
COPY requirements.txt /Portfolio-Optimizer/
RUN pip install -r /Portfolio-Optimizer/requirements.txt

# Copy the rest of the application code
COPY . /Portfolio-Optimizer/

# Set the working directory back to the main application directory
WORKDIR /Portfolio-Optimizer

# Command to run your application
CMD ["python", "main.py"]
