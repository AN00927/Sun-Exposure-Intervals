# TL-01 Conversion Factor

### 1. The "Good Enough" Literature Conversion Factor (From Rodríguez-Granados 2017)
The Rodríguez-Granados paper provides the exact empirical data you need for their specific setup (a Waldmann UV7002 cabinet with 21 × F79/120W-TL01 tubes). 

They measured:
*   **Absolute physical irradiance:** 4.73 mW/cm²
*   **CIE erythemal-weighted irradiance:** 0.26 mW/cm²

**The Math:**
1.  **Ratio:** 0.26 / 4.73 = **0.055** (This means exactly 5.5% of the total physical energy output of this specific cabinet is erythemally effective).
2.  **Conversion to SED:** 1 SED is defined as 100 J/m², which equals **10 mJ/cm²** of *erythemal* dose.
3.  **The Formula:** If your dosimeter reads **1.0 J/cm²** of *absolute physical* dose, the erythemal dose is `1.0 × 0.055 = 0.055 J/cm²` (or 55 mJ/cm²). 
4.  **Final Conversion:** 55 mJ/cm² ÷ 10 mJ/cm²/SED = **5.5 SED**.

> **Rough Rule of Thumb from this paper:** **1 J/cm² (absolute physical dose) ≈ 5.5 SED** *(for a Waldmann UV7002 with 21 TL-01 tubes).*

---

### 2. What is Missing to Make This Accurate for *Your* Setup?
You cannot safely apply the "5.5" factor to a different cabinet, a different number of tubes, or an older set of lamps. To get a *true* conversion factor for your specific environment, you need:

1. **Your Specific Spectral Irradiance Data (µW/cm²/nm):** You need the actual wavelength-by-wavelength output of *your* TL-01 lamps. Lamp spectrum shifts slightly as phosphors degrade, and total output drops significantly with age (tube aging). Cabinet geometry (reflectors, distance from tubes) also drastically alters the integrated dose.
2. **The CIE McKinlay-Diffey Convolution:** You must take your specific spectral data and mathematically multiply (convolve) it at every wavelength by the CIE McKinlay-Diffey erythemal action spectrum, then integrate the area under the curve. (This is the methodology confirmed by Pérez Ferriols 2014).
3. **Knowledge of Your Meter's Output:** Is the "J/cm²" your machine displays *absolute physical energy* or is the manufacturer claiming it is *already erythemal-weighted*? (Many claim the latter, but fail at it, as shown below).

---

### 3. The Critical Trap: Why You Can't Trust Standard Hand-Held Meters (Lloyd 2004)
The Lloyd (2004) paper is the most important warning in your stack. It proves that **standard hand-held UV meters are notoriously unreliable for TL-01 calibration.** 

*   **The Finding:** Calibration factors for TL-01 varied by **±60% (up to a 2.7x difference)** across 7 expert UK medical physics departments using traceable equipment. 
*   **The Cause:** 
    1. **Spectral Mismatch:** TL-01 has a very sharp, narrow peak at 311 nm. Most broadband UV meters are calibrated for a wider spectrum. If the meter's filter doesn't perfectly match the TL-01 spike, it will wildly over- or under-read.
    2. **Cosine (Angular) Error:** As Narbutt (2014) and Martin et al. point out, if the meter's sensor doesn't have a near-perfect cosine response (`f2 < 10%`), it will misread the scattered light bouncing off the cabinet walls versus the direct light from the tubes. 

*Conclusion:* If you are using a standard hand-held dosimeter to get your "J/cm²" number, that number is likely already flawed by 20% to 170% before you even apply a SED conversion.

---

### 4. Actionable Next Steps: How to Get Your Real Number
To get a legally and clinically defensible J/cm² → SED conversion factor for your specific practice, you have two paths:

**Path A: The Empirical Measurement (Gold Standard)**
1. Hire a medical physics department or specialized dosimetry lab (like those in the Lloyd or Narbutt papers).
2. Have them bring a **double-monochromator spectroradiometer** (e.g., MACAM-SR 2271 or Bentham DTMc300) to your clinic.
3. Have them measure the absolute and erythemal-weighted irradiance of your specific cabinet at the patient's skin distance.
4. Calculate your facility-specific ratio (like the 0.055 ratio above) and update it annually as the lamps age.

**Path B: The Manufacturer Data Route (Practical Compromise)**
1. Contact the manufacturer of your specific phototherapy cabinet (e.g., Waldmann, Daavlin, National Biological).
2. Request the **spectral power distribution (SPD) data** (in µW/cm²/nm) for your exact cabinet model and lamp type, ideally at a specific hour mark (e.g., "at 500 hours of use").
3. Use a spreadsheet to convolve that provided SPD data with the publicly available **CIE McKinlay-Diffey erythemal action spectrum** (from Diffey's 1987 paper or IPEM Report 76).
4. Integrate the result to find your specific erythemal yield, and derive your own J/cm² → SED multiplier.

**Summary:** You have a valid *placeholder* formula (`SED ≈ J/cm²_absolute × 5.5`) from the Rodríguez-Granados paper, but the Lloyd (2004) paper proves that relying on generic conversion factors or standard hand-held meters for TL-01 is clinically dangerous. You must either measure your specific cabinet's spectrum or obtain the manufacturer's spectroradiometric data for your exact model to do the CIE-weighted math yourself.

TO BE CLEAR: TL-01 is EXTREMELY volatile, and honestly should not be used in a high dependency/effect area to make large conclusions. Instead, sun based exposure is optimal. 
