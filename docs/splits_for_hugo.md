# Mail to Hugo: test boxes and his VAE split

Written 13 October 2026 from the draw in `derived/test_boxes/` (rule `docs/test_box_rule.md`,
commit a23b627; code b972889) and the split in `derived/hugo_vae_split_2425/` (code 52e1088).
For Ruben to paste into a mail.

---

Subject: Test boxes and your VAE split — for a quick check

Hi Hugo,

As agreed on 13 October, I fixed the test set for the 2025–26 data and reproduced your VAE split.
Could you have a quick look and tell me whether you agree?

**1. Test boxes (2025–26).** I listed all 455 ways to choose 3 of the 15 boxes and kept the sets
whose counts of non-consumable fruit (defective ≥ 2), browning ≥ 2 without cavity, browning 3, and
any cavity are within ±30 % of what is expected for those 3 boxes (expected = season total × fruit
in the set / 433 scanned fruit). 120 sets qualified. I drew one at random with seed 369: boxes
**F, J and M** (87 fruit). Their counts beside the expected counts:

| Count | Test boxes | Expected | Allowed (±30 %) |
|---|---|---|---|
| non-consumable | 35 | 36.0 | 25.2 to 46.8 |
| browning ≥ 2 without cavity | 18 | 14.7 | 10.3 to 19.1 |
| browning 3 | 19 | 20.1 | 14.1 to 26.1 |
| any cavity | 22 | 23.1 | 16.2 to 30.0 |

Every box occurs in at least 18 of the qualifying sets, so no box was ruled out in advance. These
boxes stay untouched until the analysis is frozen; all choices are made on the other 12 boxes.

**2. Cross-validation folds.** The other 12 boxes are split into 4 folds of 3 boxes by the same
rule (each fold within ±30 % of the 12-box totals; 186 of the 15,400 possible splits qualify). The
cross-validation is repeated with 5 of them, drawn with seed 369:

| Repeat | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|
| 1 | A B I | C G N | D E O | H K L |
| 2 | A C E | B G L | D I N | H K O |
| 3 | A E H | B G L | C K O | D I N |
| 4 | A H N | B D L | C E I | G K O |
| 5 | A G K | B C I | D N O | E H L |

**3. Your VAE split (2024–25).** With `split_train_val_test`, validation 0.1, test 0.1, seed 369,
over the rows of `labels_conference_pear_2425.csv` in file order, I get 528 / 66 / 66. The
validation and test fruit are:

Validation (66): A20, A22, A23, A25, A33, A36, A42, B14, B15, B20, B37, B53, C01, C06, C14, C24,
C41, C47, C59, D04, D32, D35, D52, E15, E25, E26, E39, E55, F11, F23, F42, F44, F59, G05, G12,
G17, G21, G26, G34, G37, G50, G_opt03, G_opt15, H13, H31, H33, H34, H35, H40, H41, H53, I05, I20,
I22, I28, I44, I45, I_opt05, I_opt09, I_opt13, J08, J14, J24, J27, J32, J37.

Test (66): A01, A02, A06, A10, A11, A26, B02, B11, B12, B16, B21, B38, B43, B59, C03, C27, C44,
C52, D05, D20, D26, D44, D49, E12, E38, E42, E48, E56, E58, E60, F18, F24, F33, F41, F52, G04,
G18, G25, G35, G42, G45, G58, G_opt12, G_opt14, G_opt28, H04, H08, H25, H36, H45, H46, H57, I10,
I25, I38, I43, I48, I50, I56, I_opt14, I_opt16, J28, J30, J34, J47, J52.

Is this the same as your split? It only matches if your `labels.csv` has the same rows in the same
order as `labels_conference_pear_2425.csv`. I will not use your 528 training fruit in the analysis
(only to compare their reconstruction error with unseen fruit); your 66 test fruit join my final
test set as one batch; your 66 validation fruit are on the training side.

Thanks, Ruben
