import React, { useState } from 'react';
import useFetch from "../hooks/useFetch";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Rectangle,
} from 'recharts';

const monthNames = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
];


const MyCharts = ({ selectedYear }) => {
  const { data } = useFetch('https://tech4good-backend-production.up.railway.app/api/data');
  const filteredData = data ? data.filter(item => item.Any === selectedYear) : [];
  const slicedData = filteredData.slice(0, 12);

  return (
    <>
    <h1 className='text-4xl text-[#1C315E] text-center font-semibold p-8'>Tendencia Anual de Precipitación, Temperatura y SPI</h1>
        <p className='text-[#227C70]'>Este gráfico muestra la tendencia anual de la precipitación, temperatura y el Índice de Severidad de la Sequía (SPI) a lo largo de los años.</p>
    <div style={{ height: '500px', position: 'relative' }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={slicedData}
          margin={{
            top: 20,
            right: 20,
            bottom: 20,
            left: 20
          }}
        >
          <CartesianGrid stroke="#f5f5f5" />
          <XAxis
            dataKey={(item) => `${item.Mes}-${item.Any}`}
            tickFormatter={(value) => {
              const [month, year] = value.split('-');
              return `${monthNames[Number(month) - 1]} ${year}`;
            }}
          />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="Precipitacion" stroke="#8884d8" name="Precipitation" />
          <Line type="monotone" dataKey="Temperatura" stroke="#82ca9d" name="Temperature" />
          <Line type="monotone" dataKey="SPI" stroke="red" name="SPI" />
        </LineChart>
      </ResponsiveContainer>

    </div>
    </>
  );
};

const Charts = () => {
  const [selectedYear, setSelectedYear] = useState(2022); // Set the initial selected year here

  const handleYearChange = (event) => {
    const year = parseInt(event.target.value);
    setSelectedYear(year);
  };

  return (
    <div className='py-2'>
      <label htmlFor="year" className="mr-2 text-primary">Año:</label>
      <select
        id="year"
        className="appearance-none bg-white border border-secondary rounded-md py-2 px-4 focus:outline-none focus:border-primary"
        onChange={handleYearChange}
        value={selectedYear}
      >
        {Array.from({ length: 23 }, (_, index) => 2000 + index).map(year => (
          <option key={year} value={year}>
            {year}
          </option>
        ))}
      </select>
      <MyCharts selectedYear={selectedYear} />
      <div className="mt-4 flex  align-items justify-center">
        <h4 className='text-lg font-semibold mx-2'>Rangos de SPI:</h4>
        <div className="mr-4 flex  items-center">
          <div className="w-4 h-4 bg-red-500 mr-2"></div>
          <div>-2 to -0.5: Sequera severa</div>
        </div>
        <div className="mr-4 flex  items-center">
          <div className="w-4 h-4 bg-blue-500 mr-2"></div>
          <div>-0.5 to 0.5: Tiempo normal</div>
        </div>
        <div className="mr-4 flex  items-center">
          <div className="w-4 h-4 bg-green-500 mr-2"></div>
          <div>0.5 to 2: Condiciones húmedas</div>
        </div>
    </div>
    </div>
  );
};

export default Charts;
