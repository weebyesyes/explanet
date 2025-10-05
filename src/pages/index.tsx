import React, { useState, type KeyboardEvent } from "react";
import Link from "next/link";
import styles from "../styles/pages/home.module.css";
import styless from "../styles/pages/index.module.css";

export default function Home() {
  // isRight===false => show "left" content; card at left
  // isRight===true  => show "right" content; card slid to the right
  const [isRight, setIsRight] = useState(false);

  const toggle = () => setIsRight((v) => !v);

  const handleCardKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggle();
    }
  };

  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <h1 className={styles.heroTitle}>
          A World Away: Hunting for Exoplanets with AI
        </h1>
        <p className={styless.heroSubtitle}>
          Upload mission catalogs. Get planet probabilities. Explore why the model thinks so.
        </p>
      </section>

      <section className={styles.aboutSection}>
        <div className={styles.aboutLogo} aria-hidden="true">
          <span>ABOUT</span>
          <span>EXORB</span>
        </div>
        <div className={styles.aboutContent}>
          <p>
            Exoplanets are worlds that orbit stars beyond our Sun. Many are found with the transit
            method: when a planet crosses in front of its star, the star’s brightness dips slightly.
          </p>
          <p>By measuring these tiny dips across time, we infer things like:</p>
          <ul className={styles.featureList}>
            <li>
              <span className={styles.featureLabel}>Orbital period (days):</span> time between dips
            </li>
            <li>
              <span className={styles.featureLabel}>Transit duration (hours):</span> how long the dip lasts
            </li>
            <li>
              <span className={styles.featureLabel}>Transit depth (ppm):</span> how deep the dip is (how much light is blocked)
            </li>
            <li>
              <span className={styles.featureLabel}>Planet radius (R⊕):</span> estimated from depth + star size
            </li>
            <li>
              <span className={styles.featureLabel}>Star properties:</span> temperature (Teff), surface gravity (log g), radius, brightness
            </li>
            <li>
              <span className={styles.featureLabel}>Signal quality:</span> SNR-like measures
            </li>
          </ul>
          <p>
            These are exactly the columns our AI uses to judge if a signal is a Planet, a Candidate, or a False Positive.
          </p>
        </div>
      </section>

      <section className={styles.cardSection}>
        <div className={styles.cardWrapper}>
          {/* ONE CARD that slides; content cross-fades inside */}
          <div
            className={`${styles.infoCard} ${isRight ? styles.infoCardShifted : ""}`}
            onClick={toggle}
            onKeyDown={handleCardKeyDown}
            role="button"
            tabIndex={0}
            aria-pressed={isRight}
          >
            {/* Overlay viewport so the two content layers stack perfectly */}
            <div className={styles.cardViewport}>
              {/* LEFT CONTENT (Why/What/Datasets) */}
              <div
                className={`${styles.contentLayer} ${styles.contentLeft} ${
                  isRight ? styles.hidden : styles.visible
                }`}
                aria-hidden={isRight}
              >
                <h3 className={styles.cardTitle}>Why this matters</h3>
                <p>
                  There’s far more data than people to review. An accurate, transparent AI can triage
                  large catalogs, surface the most promising candidates, and help researchers focus
                  their time—without replacing human vetting.
                </p>

                <h3 className={styles.cardTitle}>What we predict</h3>
                <ul className={styles.cardList}>
                  <li>P(planet) — a calibrated probability</li>
                  <li>Predicted class — Planet / Candidate / False Positive</li>
                  <li>Confidence tier — High / Medium / Low (based on P(planet))</li>
                  <li>A threshold flag — whether it clears the acceptance cutoff (global or per-mission)</li>
                </ul>

                <h3 className={styles.cardTitle}>Datasets we built on</h3>
                <p>
                  We trained with public data from Kepler, K2, and TESS. Each contains confirmed
                  planets, candidates, and false positives, plus the features above. Your uploads must
                  label their mission as Kepler, K2, or TESS so we can apply the correct preprocessing.
                </p>
              </div>

              {/* RIGHT CONTENT (How it works) */}
              <div
                className={`${styles.contentLayer} ${styles.contentRight} ${
                  isRight ? styles.visible : styles.hidden
                }`}
                aria-hidden={!isRight}
              >
                <h3 className={styles.cardTitle} style={{ textAlign: "center" }}>How the app works</h3>
                <ol className={styles.cardList}>
                  <li>
                    <strong>1) Data · Validator</strong>
                    <span>
                      {" "}
                      Upload a CSV (or ZIP with a CSV). We check required columns (id, mission), warn on
                      odd values, flag duplicate IDs and “leaky” columns (anything that could reveal the
                      label), and show a clear Ready / Needs fixes / Cannot score status with issue details.
                    </span>
                  </li>
                  <li>
                    <strong>2) Predict &amp; Results</strong>
                    <span>
                      {" "}
                      We send your file to /api/predict, run the ensemble, and return a paginated table +
                      a downloadable CSV. You’ll see the threshold used (and per-mission overrides if
                      enabled).
                    </span>
                  </li>
                  <li>
                    <strong>3) Visualize</strong>
                    <span>
                      {" "}
                      Once /api/visualize finishes, you’ll get an interactive dashboard: QC: mission mix,
                      missingness heatmap, duplicates, histograms, scatter plots, outliers; Drift: compare
                      your upload to training distributions; Tuning &amp; performance: score histogram, mission
                      acceptance, PR/ROC/calibration curves, confusion snapshots, top candidates, feature
                      importance, and (optionally) SHAP-style summaries.
                    </span>
                  </li>
                </ol>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className={styles.underTheHoodSection}>
        <h2 className={styles.sectionHeading}>Under the hood (AI, simply explained)</h2>
        <ul className={styles.underTheHoodList}>
          <li>
            <span className={styles.underTheHoodLabel}>Inference only in the app.</span> Training happened
            offline; the web just runs scoring.
          </li>
          <li>
            <span className={styles.underTheHoodLabel}>Preprocessing that mirrors training.</span> We
            harmonize features (period, duration, depth, radii, star context, etc.), add physics-aware
            ratios (e.g., depth vs. expected from radii), and one-hot the mission.
          </li>
          <li>
            <span className={styles.underTheHoodLabel}>Ensemble of gradient-boosted trees.</span> We load
            LightGBM/CatBoost/XGBoost models.
          </li>
          <li>
            <span className={styles.underTheHoodLabel}>Two-stage logic:</span> Stage A: Planet vs. Not-Planet
            · Stage B: if Not-Planet, Candidate vs. False Positive → produces a 3-class probability vector.
          </li>
          <li>
            <span className={styles.underTheHoodLabel}>Blending &amp; meta-learning:</span> We average across
            seeds/folds/families; if present, a logistic meta-blender learns the best combo.
          </li>
          <li>
            <span className={styles.underTheHoodLabel}>Calibration:</span> A Platt calibrator makes P(planet)
            more honest.
          </li>
          <li>
            <span className={styles.underTheHoodLabel}>Thresholds:</span> Use a recommended cutoff, Top-N, or
            per-mission thresholds (Kepler/TESS/K2 can differ).
          </li>
          <li>
            <span className={styles.underTheHoodLabel}>Strict missions:</span> uploads must say Kepler, K2, or
            TESS exactly—no guessing.
          </li>
        </ul>
      </section>

      <section className={styles.gettingStartedSection}>
        <h2 className={styles.sectionHeading}>Getting started</h2>
        <ul className={styles.gettingStartedList}>
          <li>
            Go to{" "}
            <Link href="/data" className={styles.navigationLink}>
              <strong>Data (Validator)</strong>
            </Link>{" "}
            and upload your CSV/ZIP.
          </li>
          <li>If status is Ready (or Needs fixes with acceptable warnings), continue.</li>
          <li>
            Open{" "}
            <Link href="/predict_and_results" className={styles.navigationLink}>
              <strong>Predict &amp; Results</strong>
            </Link>{" "}
            to score and download the table.
          </li>
          <li>
            Explore{" "}
            <Link href="/visualize" className={styles.navigationLink}>
              <strong>Visualize</strong>
            </Link>{" "}
            to QC distributions, compare to training, and tune thresholds.
          </li>
        </ul>
      </section>

      <section className={styles.challengeSection}>
        <h2 className={styles.sectionHeading}>About the challenge</h2>
        <p>
          This project was built for the 2025 NASA Space Apps Challenge (Advanced). The goal: use open NASA
          datasets (Kepler, K2, TESS) and AI/ML to classify exoplanet signals, plus a web UI so people can
          upload new data, see predictions, and understand why.
        </p>
      </section>
    </div>
  );
}
