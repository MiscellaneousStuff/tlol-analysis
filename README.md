# Assumptions

- Up to 3 min
  - Minion names are restricted to only those which appear in the
    first 3 mins (1-hot encoding of their names).
- Player controlled champ is Ezreal (All of the action spec is
  strongly tailored around this. Melee champs are really difficult
  to implement. Ranged champs are easier and if active spell gets
  correctly implemented will become trivial to implement)

# Issues

- Auto attack inference from missiles is unreliable, especially at 16x replay speed.
  - Not all auto attacks are being correctly scraped (probably needs to be 2x or 4x at most :/).
    Or activespell needs to be correctly fixed within T_T-Pandoras-Box. This would be far more reliable
    and allow for much higher throughput.
  - In summary: current detected auto attacks have high true postive accuracy but have low recall.