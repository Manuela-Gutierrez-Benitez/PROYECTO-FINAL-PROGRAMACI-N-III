from model_ml.model import predict_label

def get_user_inputs():
    print("=== Prediccion de Cultivo ===")

    try:
        K = float(input("Valor de K: "))
        ph = float(input("Valor de pH: "))
        humidity = float(input("Humedad (%): "))
    except ValueError:
        print("\nError: los valores deben ser numericos.")
        return

    # Llamar al modelo
    result = predict_label(K, ph, humidity)
    print(f"\nCultivo predicho: {result}")


def menu():
    while True:
        print("========== MENÚ ==========")
        print("1. Realizar una prediccion")
        print("2. Salir")
        print("==========================")

        opc = input("Seleccione una opcion: ")

        if opc == "1":
            get_user_inputs()

        elif opc == "2":
            print("\nSaliendo del programa...")
            break

        else:
            print("\n Opción invalida, intenta otra vez.\n")