import { Outlet } from "react-router-dom";
import Footer from "../components/Footer";
import { Header } from "../components/Header";

export default function RootLayout() {
  return (
    <>
       <Header />
      <main className='content'>
        <Outlet />
      </main>
      <Footer />
    </>
  );
}
