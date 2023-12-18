# TLoL Dataset Analysis (Patch 13.23: Ezreal 60k)

## Assumptions

- Up to 3 min
  - Minion names are restricted to only those which appear in the
    first 3 mins (1-hot encoding of their names).
- Player controlled champ is Ezreal (All of the action spec is
  strongly tailored around this. Melee champs are really difficult
  to implement. Ranged champs are easier and if active spell gets
  correctly implemented will become trivial to implement)

## Issues

- Auto attack inference from missiles is unreliable, especially at 16x replay speed.
  - Not all auto attacks are being correctly scraped (probably needs to be 2x or 4x at most :/).
    Or activespell needs to be correctly fixed within T_T-Pandoras-Box. This would be far more reliable
    and allow for much higher throughput.
  - In summary: current detected auto attacks have high true postive accuracy but have low recall.
- Ezreal Q Spell Usage
  - Discretised aim of the spell is problematic. Only covers +/- 400 units but ezreal q range is 1100 units.
    This means that the more fine tuned aiming of the spell is just off. This extends also to the W as well.

## Conversion Considerations

- When encoding labels across an entire dataset during bulk conversion for ML datasets, the indexing
  of the embeddings needs to be consistent. There are two ways to achieve this:
  1. Get the original listing of whatever it is, so for spell names, get every single internal spell
     name from data dragon for every spell in League of Legends as a list and index into this
     (This is by far the best method. Need to account for inter-patch changes such as spells being
      changed or new champions like Hwei which aren't currently accounted for)
  2. Use hasing to compensate by hashing any found spell name, create a build time list of strings
     for that particular string type (like missile name) and then index into that custom list of strings.
     This also works but need to be careful during builk conversion run time. As long as it's kept consistent
     this is also fine and works.

# Data Verification Results

- Auto attack distances between Ezreal and targets are often much higher than AA range, this indicates
  either a post-processing data handling bug, or something during runtime when scraping? Much more likely
  to be in the post-process data handling part though.