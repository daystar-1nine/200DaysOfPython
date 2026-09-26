import numpy as np

def sigmoid(x):
    # Clip x to avoid overflow in exp
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class ScratchLSTMCell:
    """
    A pure NumPy implementation of an LSTM cell's forward equations.
    This demonstrates the internal mechanics of how long-term dependencies are handled.
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights and biases (simplified, combining W_i and W_h for each gate)
        # We concatenate x_t and h_{t-1} so the weight matrix has size (input_size + hidden_size, hidden_size)
        
        # Forget gate
        self.W_f = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
        self.b_f = np.ones((1, hidden_size)) # Initialize forget gate bias to 1 is a good practice
        
        # Input gate
        self.W_i = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
        self.b_i = np.zeros((1, hidden_size))
        
        # Candidate cell state (g or C_tilde)
        self.W_c = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
        self.b_c = np.zeros((1, hidden_size))
        
        # Output gate
        self.W_o = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
        self.b_o = np.zeros((1, hidden_size))
        
    def forward(self, x_t, h_prev, c_prev):
        """
        Computes one time step of the LSTM.
        
        x_t: Input at time t (batch_size, input_size)
        h_prev: Hidden state at time t-1 (batch_size, hidden_size)
        c_prev: Cell state at time t-1 (batch_size, hidden_size)
        
        Returns:
        h_t: Hidden state at time t
        c_t: Cell state at time t
        cache: Tuple of intermediate values for backprop (if needed)
        """
        # Concatenate input and previous hidden state
        # shape: (batch_size, input_size + hidden_size)
        concat = np.hstack((x_t, h_prev))
        
        # 1. Forget gate: decides what information to throw away from the cell state
        f_t = sigmoid(np.dot(concat, self.W_f) + self.b_f)
        
        # 2. Input gate: decides which values we'll update
        i_t = sigmoid(np.dot(concat, self.W_i) + self.b_i)
        
        # 3. Candidate cell state: creates a vector of new candidate values
        c_tilde = tanh(np.dot(concat, self.W_c) + self.b_c)
        
        # 4. Update the cell state
        # Old state multiplied by f_t (forgetting), plus new candidate values scaled by i_t (input)
        c_t = f_t * c_prev + i_t * c_tilde
        
        # 5. Output gate: decides what to output based on the cell state
        o_t = sigmoid(np.dot(concat, self.W_o) + self.b_o)
        
        # 6. Final hidden state
        h_t = o_t * tanh(c_t)
        
        cache = (x_t, h_prev, c_prev, f_t, i_t, c_tilde, c_t, o_t, h_t)
        return h_t, c_t, cache

if __name__ == "__main__":
    # Test the Scratch LSTM Cell
    np.random.seed(42)
    batch_size = 2
    input_dim = 10
    hidden_dim = 16
    
    lstm_cell = ScratchLSTMCell(input_dim, hidden_dim)
    
    # Dummy inputs
    x_t = np.random.randn(batch_size, input_dim)
    h_prev = np.zeros((batch_size, hidden_dim))
    c_prev = np.zeros((batch_size, hidden_dim))
    
    h_t, c_t, _ = lstm_cell.forward(x_t, h_prev, c_prev)
    
    print("LSTM Pure NumPy Forward Pass Validation:")
    print(f"Input shape: {x_t.shape}")
    print(f"Previous Hidden shape: {h_prev.shape}")
    print(f"Previous Cell shape: {c_prev.shape}")
    print(f"New Hidden State shape: {h_t.shape}")
    print(f"New Cell State shape: {c_t.shape}")
    print(f"Sample Hidden State values (batch 0, first 5 dims): {h_t[0, :5]}")
