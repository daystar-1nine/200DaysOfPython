# MiniGPT: Decoding Strategy Benchmark
**Prompt:** `MENENIUS:`

| Strategy | Temperature | k | p | Repetition (3-gram) | Distinct-1 | Distinct-2 | Qualitative Behavior |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Greedy** | 0.0 | - | - | 0.3889 | 0.2632 | 0.4865 | Deterministic argmax; prone to looping repetitions |
| **Pure Sampling** | 1.0 | - | - | 0.0000 | 1.0000 | 1.0000 | Unconstrained categorical sampling across full vocab |
| **Top-k (k=20)** | 0.8 | 20 | - | 0.0000 | 1.0000 | 1.0000 | Truncates to top 20 candidate tokens; eliminates low-prob tail |
| **Top-p (p=0.9)** | 0.8 | - | 0.9 | 0.0000 | 1.0000 | 1.0000 | Nucleus sampling over dynamic 90% cumulative probability mass |
| **Top-k + Top-p** | 0.8 | 20 | 0.9 | 0.0000 | 0.9677 | 1.0000 | Combined top-k and nucleus thresholding for peak coherence |

## Generated Samples by Decoding Strategy

### Strategy: Greedy
```text
MENENIUS:
Whe athe athe the athe se sande the the sis the the the sthe s sthe sthe s the sthe s aind the the the the the sthe s the sthe s sthe sthe s sthe sthe
```

### Strategy: Pure Sampling
```text
MENENIUS:
You, thas deabess chasenis lerineshthers more totway's wil sprake! Yourst
Whend usNat allugh n toat eve crongre?

First'rst Citizey Rel:
Wha you's ine
```

### Strategy: Top-k (k=20)
```text
MENENIUS:
Wowh sorene nous, enele athe rust arersts, you what hend sother bere
Whats the seand sthe malicts ans of tonaile,, thin sthel atll mant an eing prooon
```

### Strategy: Top-p (p=0.9)
```text
MENENIUS:
The are sisther the moll ail counced f ciouns. 

First Citizen:
Can:
We hith cous at is piolit tonches hang bof as. Whares may helit ande you,
Fit Cit
```

### Strategy: Top-k + Top-p
```text
MENENIUS:
The the be arese and anne forsthens athes athe don ber factiunes,
Whe he ban you dou mas hidigot oer oods oowhe he shat his,
Thele ackn wen senothe st
```

