FROM pytorch/pytorch

CMD ["bash"]

# Install Node.js 8 and npm 5
RUN apt-get update
RUN apt-get -y install curl gnupg
RUN curl -sL https://deb.nodesource.com/setup_11.x  | bash -
RUN apt-get -y install nodejs
RUN apt-get -y install wget

RUN mkdir -p /result/inference_img_gray
RUN mkdir -p /result/inference_img_rgb
RUN mkdir -p /result/inference_img_crop

WORKDIR /workspace

RUN pip install easydict

COPY package.json .
RUN npm install
RUN wget --no-check-certificate -O deeplab-resnet.pth.tar "https://onedrive.live.com/download?cid=02EFDB25A9A647DA&resid=2EFDB25A9A647DA%2115315&authkey=AFT5YvKdlxufOpc"
RUN wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
RUN mkdir /workspace/images

COPY . .
EXPOSE 80
ENTRYPOINT npm start 