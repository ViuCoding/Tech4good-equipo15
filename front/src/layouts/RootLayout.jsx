import { Outlet } from "react-router-dom";

export default function RootLayout() {
  return (
    <>
      <div>NAVBAR</div>
      <main className='content'>
        <Outlet />
      </main>
      <footer>FOOTER</footer>
    </>
  );
}
