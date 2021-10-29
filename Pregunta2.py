from MLP import Layer, MLP

def activation(x):
    return 1 if x >= 0 else 0

#Vector de entrada (Sin sesgo)
input_vector = [1.75, -4, 72]

#Capas de la red
layer1 = Layer(5, 3, activation)
layer2 = Layer(4, 3, activation)
layer3 = Layer(3, 3, activation)

#Creación de MLP
classifier = MLP([layer1, layer2, layer3])

#Output para el vector dado
#(Como no estamos usando la clase Dataset, se debe
# agregar manualmente el término de sesgo)
print(classifier.output([1] + input_vector))