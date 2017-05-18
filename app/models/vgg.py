class VGG:
    def __init__(self, graph_def):
        try:
            with open("vgg/tensorflow-vgg16/vgg16.tfmodel", mode='rb') as file:
                file_content = file.read()
                graph_def.ParseFromString(file_content)
                file.close()
        except FileNotFoundError as e:
            print('Cannot find VGG-16 model. Training is stopped')
            exit()