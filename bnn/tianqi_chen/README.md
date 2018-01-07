# Results from Prior Work

(Edit: these are old, but basically the results I see reaffirm that MOMSGD is
better than SGHMC when *both* algorithms are sufficiently tuned.)

- Best SGD Valid: wd 0.0001, eta 0.5, 0.02414
- Best SGD Test: wd 0.0001, eta 0.5, 0.02144

- Best MOMSGD Valid: wd 0.00001, eta 0.5, 0.01676
- Best MOMSGD Test: wd 0.00001, eta 0.5, 0.01586

- Best SGLD Valid: eta 0.5, 0.0255
- Best SGLD Test: eta 0.5, 0.02276

- Best SGHMC Valid: eta 0.1, 0.01828
- Best SGHMC Test: eta 0.1, 0.01648

SGHMC gets 0.01648 which very closely matches the result in the SGHMC paper. But
momentum with SGD is just as good, if not better (with 0.01586).

So ... it seems like SGHMC is not really that much better than momentum with
SGD. How should we spin this?
