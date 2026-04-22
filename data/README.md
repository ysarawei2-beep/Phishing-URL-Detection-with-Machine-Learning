# Data dictionary

This project uses the Kaggle dataset

> **Phishing Dataset for Machine Learning: Feature Evaluation**
> by *Shashwat Tiwari*
> <https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning>

Original source: Tan, Choon Lin (2018).
*Phishing Dataset for Machine Learning: Feature Evaluation*.
Mendeley Data, V1. DOI: `10.17632/h3cgnj8hft.1`.

**License:** CC BY 4.0. You must credit the original authors when
redistributing the dataset or derived artefacts.

---

## Files

| File | Purpose |
| --- | --- |
| `raw/Phishing_Legitimate_full.csv` | Full dataset (10 000 rows × 50 cols). Place it here after downloading from Kaggle. |
| `sample/sample_phishing.csv` | 200-row stratified sample bundled with the repo so the pipeline can run without the Kaggle download. |
| `processed/train.csv` | Training split produced by `src/training/train.py`. |
| `processed/val.csv` | Validation split. |
| `processed/test.csv` | Test split. |

---

## Schema

| Column | Type | Description |
| --- | --- | --- |
| `id` | int | Row identifier (dropped before training). |
| `NumDots` | int | Number of “.” in the URL. |
| `SubdomainLevel` | int | Depth of the subdomain hierarchy. |
| `PathLevel` | int | Depth of the URL path (slashes). |
| `UrlLength` | int | Total character length of the URL. |
| `NumDash` | int | Number of “-” in the URL. |
| `NumDashInHostname` | int | Number of “-” in the hostname. |
| `AtSymbol` | 0/1 | Presence of “@”. |
| `TildeSymbol` | 0/1 | Presence of “~”. |
| `NumUnderscore` | int | Number of “_” in the URL. |
| `NumPercent` | int | Number of “%” in the URL. |
| `NumQueryComponents` | int | Count of query parameters. |
| `NumAmpersand` | int | Count of “&”. |
| `NumHash` | int | Count of “#”. |
| `NumNumericChars` | int | Count of digits. |
| `NoHttps` | 0/1 | 1 if scheme is not HTTPS. |
| `RandomString` | 0/1 | Heuristic flag for random-looking tokens. |
| `IpAddress` | 0/1 | Hostname is a bare IPv4. |
| `DomainInSubdomains` | 0/1 | Domain name appears inside the subdomain. |
| `DomainInPaths` | 0/1 | Domain name appears inside the URL path. |
| `HttpsInHostname` | 0/1 | Literal “https” appears inside the hostname. |
| `HostnameLength` | int | Length of the hostname. |
| `PathLength` | int | Length of the URL path. |
| `QueryLength` | int | Length of the query string. |
| `DoubleSlashInPath` | 0/1 | “//” appears in the path. |
| `NumSensitiveWords` | int | Count of phishing-keyword hits (login, secure, …). |
| `EmbeddedBrandName` | 0/1 | A well-known brand is embedded in the URL. |
| `PctExtHyperlinks` | float | % of hyperlinks on the page pointing to an external domain. |
| `PctExtResourceUrls` | float | % of resource URLs (images, scripts) from external domains. |
| `ExtFavicon` | 0/1 | Favicon loaded from an external domain. |
| `InsecureForms` | 0/1 | Any `<form action="http://…">`. |
| `RelativeFormAction` | 0/1 | Form with a relative action. |
| `ExtFormAction` | 0/1 | Form posts to an external domain. |
| `AbnormalFormAction` | 0/1 | Form action is empty / unusual. |
| `PctNullSelfRedirectHyperlinks` | float | % of `href="#"` or self-redirect hyperlinks. |
| `FrequentDomainNameMismatch` | 0/1 | Most-referenced domain differs from hostname. |
| `FakeLinkInStatusBar` | 0/1 | JS code overwriting the browser status bar. |
| `RightClickDisabled` | 0/1 | Right-click disabled via JS. |
| `PopUpWindow` | 0/1 | Page opens pop-up windows. |
| `SubmitInfoToEmail` | 0/1 | `mailto:` form action. |
| `IframeOrFrame` | 0/1 | Hidden iframes / frames. |
| `MissingTitle` | 0/1 | `<title>` tag missing. |
| `ImagesOnlyInForm` | 0/1 | Form contains only images (click-map phishing). |
| `SubdomainLevelRT` | –1/0/1 | Runtime risk: subdomain depth (safe / unknown / suspicious). |
| `UrlLengthRT` | –1/0/1 | Runtime risk: URL length. |
| `PctExtResourceUrlsRT` | –1/0/1 | Runtime risk: external resource fraction. |
| `AbnormalExtFormActionR` | –1/0/1 | Runtime risk: abnormal form action. |
| `ExtMetaScriptLinkRT` | –1/0/1 | Runtime risk: external meta / script / link. |
| `PctExtNullSelfRedirectHyperlinksRT` | –1/0/1 | Runtime risk: null / self-redirect hyperlinks. |
| `CLASS_LABEL` | 0/1 | **Target.** 0 = legitimate, 1 = phishing. |

---

## Engineered features added by this project

| Column | Formula |
| --- | --- |
| `PathToUrlRatio` | `PathLength / UrlLength` |
| `HostnameToUrlRatio` | `HostnameLength / UrlLength` |
| `DigitToUrlRatio` | `NumNumericChars / UrlLength` |

These capture the proportion of the URL dedicated to the path,
hostname, and digits respectively — useful because absolute
lengths can be misleading.

---

## Class distribution

| Class | Count | Percentage |
| --- | --- | --- |
| 0 (legitimate) | 5000 | 50 % |
| 1 (phishing) | 5000 | 50 % |

The dataset is perfectly balanced, so class imbalance is **not** a
concern for this project.

---

## Data quality

* No missing values (verified on load).
* No duplicate rows in the raw CSV.
* All feature columns are numeric (int / float).
* The `id` column is unique per row and is dropped before training.
