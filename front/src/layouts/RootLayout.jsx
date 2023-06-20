import { Outlet } from "react-router-dom";
import Footer from "../components/Footer";

export default function RootLayout() {
  return (
    <>
      <div>NAVBAR</div>
      <main className='content'>
        <Outlet />
      </main>
      <Footer />
    </>
  );
}
