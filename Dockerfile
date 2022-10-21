FROM python:3.9

WORKDIR /MA_Demo

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py ./
COPY GradCam_v3.py ./
COPY misc_functions.py ./
#--server.port 8503
CMD ["streamlit", "run", "./MA_Demo/main.py", "--server.port 8503"]
EXPOSE 8503

#to build image from dockerfile
#docker build -t membrane-analysis-demo .

#run
#docker run -p 8503:8503 membrane-analysis-demo streamlit run main.py --server.port 8503