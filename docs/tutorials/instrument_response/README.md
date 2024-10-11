# TradDet

The "traditional detector" corresponds to

```
det = ToyTracker2D(material = 'Ge',
                   layer_length = 10*u.m, 
                   layer_positions = np.append(300, np.arange(0,10,1))*u.cm, 
                   layer_thickness = 1*u.cm, 
                   energy_resolution = 0.03,
                   energy_threshold = 20*u.keV)
```

Using the SimpleTraditionalReconstructor and commit b5e2ebb

response_energy_onaxis_traddet.h5 was generated using an spectral index of -1.
