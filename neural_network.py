import numpy as np
import os


class NeuralNetwork:

    # ==========================================
    # 1. INITIALIZATION
    # ==========================================

    def __init__(self, input_size, hidden_size, output_size):

        if min(input_size, hidden_size, output_size) <= 0:
            raise ValueError("Semua ukuran layer harus lebih dari 0")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Xavier initialization
        limit1 = np.sqrt(6.0 / (input_size + hidden_size))
        limit2 = np.sqrt(6.0 / (hidden_size + output_size))

        self.w1 = np.random.uniform(
            -limit1, limit1,
            (input_size, hidden_size)
        )

        self.w2 = np.random.uniform(
            -limit2, limit2,
            (hidden_size, output_size)
        )

        # Bias initialization
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(output_size)

        # Forward propagation cache
        self.last_hidden = np.zeros(hidden_size)
        self.last_output = np.zeros(output_size)

        # Training metadata
        self.best_fitness = 0.0


    # ==========================================
    # 2. ACTIVATION FUNCTION
    # ==========================================

    @staticmethod
    def activation(x):
        return np.tanh(x)


    # ==========================================
    # 3. FORWARD PROPAGATION
    # ==========================================

    def forward(self, inputs):

        inputs = np.asarray(inputs, dtype=np.float64)

        if inputs.shape != (self.input_size,):
            raise ValueError(
                f"Expected input shape ({self.input_size},), "
                f"received {inputs.shape}"
            )

        if not np.all(np.isfinite(inputs)):
            raise ValueError("Input mengandung NaN atau Infinity")

        # Input -> Hidden
        hidden_z = np.dot(inputs, self.w1) + self.b1
        self.last_hidden = self.activation(hidden_z)

        # Hidden -> Output
        output_z = np.dot(self.last_hidden, self.w2) + self.b2
        self.last_output = self.activation(output_z)

        return self.last_output.copy()


    # ==========================================
    # 4. MUTATION
    # ==========================================

    def mutate(self, rate=0.25, scale=0.5):

        if not 0.0 <= rate <= 1.0:
            raise ValueError("Mutation rate harus antara 0 dan 1")

        if scale < 0.0:
            raise ValueError("Mutation scale tidak boleh negatif")

        parameters = [self.w1, self.w2, self.b1, self.b2]

        for parameter in parameters:

            # Random mutation mask
            mask = np.random.random(parameter.shape) < rate

            # Gaussian mutation
            noise = np.random.normal(
                loc=0.0,
                scale=scale,
                size=parameter.shape
            )

            parameter += mask * noise

        return self


    # ==========================================
    # 5. COPY NEURAL NETWORK
    # ==========================================

    def copy(self):

        nn = NeuralNetwork(
            self.input_size,
            self.hidden_size,
            self.output_size
        )

        nn.w1 = self.w1.copy()
        nn.w2 = self.w2.copy()

        nn.b1 = self.b1.copy()
        nn.b2 = self.b2.copy()

        nn.best_fitness = self.best_fitness

        return nn


    # ==========================================
    # 6. SAVE MODEL
    # ==========================================

    def save(
        self,
        filename="best_brain.npz",
        generation=1,
        best_fitness=None
    ):

        if best_fitness is not None:
            self.best_fitness = float(best_fitness)

        directory = os.path.dirname(os.path.abspath(filename))
        os.makedirs(directory, exist_ok=True)

        # Save using file handle to preserve filename
        with open(filename, "wb") as file:

            np.savez_compressed(
                file,

                # Architecture
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,

                # Weights
                w1=self.w1,
                w2=self.w2,

                # Biases
                b1=self.b1,
                b2=self.b2,

                # Metadata
                generation=generation,
                best_fitness=self.best_fitness,
                version=2
            )

        print(
            f"Model saved: {filename} "
            f"(Generation {generation}, "
            f"Fitness {self.best_fitness:.2f})"
        )


    # ==========================================
    # 7. LOAD MODEL
    # ==========================================

    @staticmethod
    def load(filename="best_brain.npz"):

        if not os.path.exists(filename):
            return None, 1

        try:

            with np.load(filename, allow_pickle=False) as data:

                w1 = data["w1"]
                w2 = data["w2"]

                input_size = w1.shape[0]
                hidden_size = w1.shape[1]
                output_size = w2.shape[1]

                # Validate weight dimensions
                if w2.shape[0] != hidden_size:
                    raise ValueError("Weight dimensions tidak cocok")

                if not np.all(np.isfinite(w1)) or not np.all(np.isfinite(w2)):
                    raise ValueError("Model memiliki bobot tidak valid")

                nn = NeuralNetwork(
                    input_size,
                    hidden_size,
                    output_size
                )

                nn.w1 = w1.copy()
                nn.w2 = w2.copy()

                # Backward compatibility with old models
                if "b1" in data:
                    nn.b1 = data["b1"].copy()

                if "b2" in data:
                    nn.b2 = data["b2"].copy()

                # Validate bias dimensions
                if nn.b1.shape != (hidden_size,):
                    raise ValueError("Hidden bias dimensions tidak cocok")

                if nn.b2.shape != (output_size,):
                    raise ValueError("Output bias dimensions tidak cocok")

                generation = (
                    int(data["generation"])
                    if "generation" in data else 1
                )

                nn.best_fitness = (
                    float(data["best_fitness"])
                    if "best_fitness" in data else 0.0
                )

                if not np.isfinite(nn.best_fitness):
                    raise ValueError("Fitness model tidak valid")

                print(
                    f"Model loaded: {filename} "
                    f"(Generation {generation}, "
                    f"Fitness {nn.best_fitness:.2f})"
                )

                return nn, generation

        except Exception as error:

            print(f"Failed to load model: {error}")
            return None, 1
