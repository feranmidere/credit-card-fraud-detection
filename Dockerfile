
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /credit-card-app

# Install build dependencies for possible native requirements
RUN apt-get update \
	&& apt-get install -y --no-install-recommends build-essential gcc \
	&& rm -rf /var/lib/apt/lists/*

# Install Python dependencies if requirements.txt is present
COPY requirements.txt ./
RUN pip install --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

RUN pip install --no-cache-dir -e .


# Create unprivileged user
RUN addgroup --system app && adduser --system --ingroup app app
USER app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]