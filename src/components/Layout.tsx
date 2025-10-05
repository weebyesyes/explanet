import { useEffect, useState } from "react";
import styles from "../styles/components/Layout.module.css";
import { useRouter } from "next/navigation";
import Image from "next/image";

const planetImages = [
  require("../../public/images/planet1.png"),
  require("../../public/images/planet2.png"),
  require("../../public/images/planet3.png"),
  require("../../public/images/planet4.png"),
]

export default function Layout({ children }: { children: React.ReactNode }) {
  const [sidebarOpened, setSidebarOpened] = useState(false);
  const [mouse, setMouse] = useState({x: 0, y: 0});
  const router = useRouter();

  function SidebarButton(props: { label: string, destination: string }) {
    return (
      <div className={styles.sidebarButton} onClick={() => router.push(props.destination)}>
        <p>{props.label}</p>
      </div>
    )
  }

  const [planets, setPlanets] = useState<any[]>([]);

  useEffect(() => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    const planets: any[] = [];
    const grids = 7;

    const usedX: number[] = [];
    const usedY: number[] = [];

    for (let i = 0; i < 4; i++) {
      let randomX = Math.floor(Math.random() * grids) - 1;
      let randomY = Math.floor(Math.random() * grids) - 1;

      while (usedX.includes(randomX) || usedY.includes(randomY)) {
        randomX = Math.floor(Math.random() * grids) - 1;
        randomY = Math.floor(Math.random() * grids) - 1;
      }

      usedX.push(randomX);
      usedY.push(randomY);

      // grids
      planets[i] = {};
      planets[i].src =  planetImages[i];
      planets[i].left = randomX * Math.floor(width / grids);
      planets[i].top = randomY * Math.floor(height / grids);
      planets[i].alt = i.toString();
    }

    setPlanets(planets);

    const handleMouseMove = (e: MouseEvent) => {
      const x = e.clientX * 17 / width;
      const y = e.clientY * 17 / height;
      setMouse({ x, y });
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [])

  return (
    <div className={styles.container}>
      <div className={styles.planets}>
        {planets.map(planet => (
          <Image
            key={`planet${planet.alt}`}
            src={planet.src}
            alt={planet.alt}
            width={320}
            style={{
              position: 'absolute',
              left: planet.left + mouse.x,
              top: planet.top + mouse.y,
              filter: 'brightness(50%)',
              zIndex: -1
            }}
          />
        ))}
      </div>

      <div className={styles.content}>
        {children}
      </div>

      {sidebarOpened && (
        <div onClick={() => setSidebarOpened(false)} className={`${styles.darkOverlay} ${sidebarOpened ? styles.darkOverlayOpened : ""}`} />
      )}

      <div className={`${styles.sidebar} ${sidebarOpened ? styles.sidebarOpened : styles.sidebarClosed}`}>
        <div className={styles.sidebarContent}>
          <Image src={require("../../public/images/logo.png")} alt="logo" width={180} className={styles.sidebarTitle} />

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