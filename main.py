import typer
import numpy as np
import keras

app = typer.Typer()

hacked_path = 'hacked_model/'


def get_model(model_path: str):
    """
    :param model_path: model path of h5 file
    :return: keras h5 model
    """
    model = keras.models.load_model(model_path)
    return model


@app.command()
def get_model_layer_name(model_path, index: int):
    """
    :param model_path: model path of h5 file
    :param index: layer number
    :return: name of layer
    """
    model = get_model(model_path)
    layer_name = model.layers[index].name
    print("Layer Name: ", layer_name)
    return layer_name


@app.command()
def get_output_layer(model_path: str):
    """
    :param model_path: model path of h5 file
    :return: final layer of model
    """
    model = get_model(model_path)
    output_layer = model.layers[-1]
    print(output_layer.bias.name)
    print(output_layer.bias.numpy())
    return output_layer


def get_output_length(model_path: str):
    """
    :param model_path: model path of h5 file
    :return: number of output layer categories
    """
    output_layer = get_output_layer(model_path)
    class_length = len(output_layer.bias.numpy())
    print(class_length)
    return class_length


@app.command()
def modify_output_bias(model_path: str, index: int):
    """
    :param model_path: model path of h5 file
    :param index: index of the classification to be manipulated
    :return: modified output bias
    """
    output_layer = get_output_layer(model_path)
    class_length = get_output_length(model_path)
    bias = []
    for i in range(class_length):
        bias.append(0)
    bias[index] = 100
    output_layer.bias.assign(bias)
    print(bias)
    # update_model(bias)


# def update_model(bias):
    # refit the model with the new bias ?
    # hacked_model.save('hacked_model/model.h5')


if __name__ == "__main__":
    app()
