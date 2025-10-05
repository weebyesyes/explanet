import { useState } from "react";
import styles from "../styles/components/Layout.module.css";
import { useRouter } from "next/navigation";

export default function Layout({ children }: { children: React.ReactNode }) {
  const [sidebarOpened, setSidebarOpened] = useState(false);
  const router = useRouter();

  function SidebarButton(props: { label: string, destination: string }) {
    return (
      <div className={styles.sidebarButton} onClick={() => router.push(props.destination)}>
        <p>{props.label}</p>
      </div>
    )
  }

  return (
    <div className={styles.container}>
      {children}

      {sidebarOpened && (
        <div className={`${styles.darkOverlay} ${sidebarOpened ? styles.darkOverlayOpened : ""}`} />
      )}

      <div className={`${styles.sidebar} ${sidebarOpened ? styles.sidebarOpened : styles.sidebarClosed}`}>
        <div className={styles.sidebarContent}>
          <p className={styles.title}>EXPLANET</p>
          <SidebarButton label="Home" destination="/" />
          <SidebarButton label="Data (Validator)" destination="data" />
          <SidebarButton label="Predict & Results" destination="predict_and_results" />
          <SidebarButton label="Visualize" destination="visualize" />
        </div>
      </div>

      <div className={`${styles.sidebarOpenButton} ${!sidebarOpened ? styles.sidebarOpenButtonClosed : ""}`} onClick={() => setSidebarOpened(sidebarOpened ? false : true)}>
        <div className={styles.sidebarOpenButtonSquare} />
        <div className={styles.sidebarOpenButtonTriangle} />
      </div>
    </div>
  );
}