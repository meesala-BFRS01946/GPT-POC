# Use the official Python image as the base
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port on which your Flask app will run (if applicable)
EXPOSE 3000

# Set any necessary environment variables (if applicable)
# ENV MY_VARIABLE=value
#ENV OPENAI_API_KEY="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"


# Define the command to run your Flask app
CMD ["python", "app.py"]

