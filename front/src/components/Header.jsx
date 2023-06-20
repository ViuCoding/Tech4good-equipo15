import { StyledContainer } from "../pages/Home";

export const Header = () => {
  return (
    <header>
      <StyledContainer>
        <div className='flex flex-row justify-between items-center p-8'>
          <div>
            <img className='h-16' src='src/assets/logo.png' alt='Logo' />
          </div>
          <ul className='flex flex-row font-semibold text-#1C315E  text-lg gap-10'>
            <li>
              <a href='#'>Precipitaciones</a>
            </li>
            <li>
              <a href='#'>Depositos de agua pluvia</a>
            </li>
            <li>
              <a href='#'>Media de temperatura mensual</a>
            </li>
          </ul>
        </div>
        <div className="flex justify-center items-center relative bg-[url('https://img1.goodfon.com/original/1600x900/0/df/barselona-ispaniya-dvorec-69.jpg')] bg-cover h-[550px] ">
          <div className='flex justify-center items-center absolute px-4 py-3 bg-gray-200/50 w-full h-[550px]'>
            <h1 className='text-[#1C315E] text-center font-bold text-[60px]'>
              La sequía es uno de los mayores problemas que afrontamos en
              Cataluña en los últimos años.
            </h1>
          </div>
        </div>
      </StyledContainer>
    </header>
  );
};
// #1C315E primario
//#227C70 secundario
//#88A47C compl
//#E6E2C3 compl2
