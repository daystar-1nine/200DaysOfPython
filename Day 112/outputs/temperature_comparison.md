# MiniGPT: Temperature Sensitivity Analysis
**Prompt:** `First Citizen:`

Temperature modifies logit sharpness: $z' = z / T$.

| Temperature | Observation | Repetition (3-gram) | Distinct-1 | Distinct-2 |
|---|---|:---:|:---:|:---:|
| **0.3** | Sharp, deterministic, repetitive | 0.0857 | 0.4054 | 0.6944 |
| **0.7** | Balanced, coherent continuation | 0.0000 | 0.8000 | 1.0000 |
| **1.0** | Balanced, coherent continuation | 0.0000 | 1.0000 | 1.0000 |
| **1.3** | High entropy, diverse, degraded syntax | 0.0000 | 0.9643 | 1.0000 |

## Generated Text Samples

### Temperature = 0.3
```text
First Citizen:
Whe the athe thand at the inde the the the theling the the ands sthe sthe the sthe the the sthe s sthe sthe sthe the s sains the and the the the the o
```

### Temperature = 0.7
```text
First Citizen:
Whery ayou you the sheralve ous thinte ano aver the s's wall thit s poreat trus the ant s theingus!

Firnst Citizen:
Del you thesenou, ssinks ant thar
```

### Temperature = 1.0
```text
First Citizen:
Thoute ay nou a. Wit withe nulorse emermakesT ane thoutecep ep ray
buritetllly ourt him emars tensus. Mat makim
FAnd tirs resend ba
Inckan oue me ilvi
```

### Temperature = 1.3
```text
First Citizen:
You vinlpstt tioh derAn'c clid, cnin mpp aley hoe
Datuou 'fat; the nele the pcomanrs, iHanat yu havis,
Caive hic! T,--cakeny, Thau you mAif catts,
Ou 
```

