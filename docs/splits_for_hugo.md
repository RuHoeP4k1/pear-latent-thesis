# Mail to Hugo: test boxes and his VAE split

Written 13 October 2026 from the draw in `derived/test_boxes/` (rule `docs/test_box_rule.md`,
commit a23b627; code b972889) and the split in `derived/hugo_vae_split_2425/` (code 52e1088).
Revised 20 October 2026 (Mac): the split was recomputed with Hugo's function, PyTorch 2.14.1, and
equals the laptop file of 13 October; the label file has the same SHA-256. The fruit lists are now
an attachment: `derived/mail_hugo_2026-10-20/vae_split_check_2425.xlsx` (tabs settings, train,
validation, test; columns `filename`, `row_in_csv` (0-based); no labels; not in git; written with
`uv run --with xlsxwriter` from the CSV beside it). For Ruben to paste into a mail.

---

Subject: Test boxes, your VAE split, and Thursday's meeting

Hi Hugo,

Sorry for the slow progress since last week: I had laptop problems that held up my work quite a
bit. That is solved now, and the encoding with your VAE is finished.

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

**3. Your VAE split (2024–25).** This is purely a safety check, so that the research rests on the
right groups: I would like to be sure that the fruit I treat as unseen by your VAE really are.
The attached `vae_split_check_2425.xlsx` has one tab per group (train, validation, test) with
the file names, so you can compare them directly with yours; the split I get is
528 train / 66 validation / 66 test. Settings (also in the first tab):

- your `utils/splits.split_train_val_test` from your repository at commit 183112f,
  `val_split = 0.1`, `test_split = 0.1`, `random_state = 369`;
- the dataset is the 660 rows of `labels_conference_pear_2425.csv` in file order, as
  `CsvVolumeDataset` reads them (`row_in_csv` in the attachment is that order, starting at 0);
- PyTorch 2.14.1 on CPU.

It only matches if your `labels.csv` has the same rows in the same order, and if your PyTorch
version draws the same permutation. Could you check whether these are your validation and test
fruit? If it is easier, a list of your 66 test files is enough for me to compare.

How I use the groups: your 528 training fruit are left out of the analysis (except to compare their
reconstruction error with unseen fruit); your 66 test fruit join my final test set as one batch;
your 66 validation fruit stay on the training side, since they were used to choose the checkpoint.

**4. Next meeting.** Could our meeting next Thursday (29 October) start one hour later? I have
another mandatory session at that time.

Thanks, Ruben

Attachment: `vae_split_check_2425.xlsx`
