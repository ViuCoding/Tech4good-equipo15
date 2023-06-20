import React from 'react';
// import data from "../pages/data"
import {
  ComposedChart,
  Line,
  Area,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Scatter,
  ReferenceLine,
  ZAxis,
  ResponsiveContainer,
} from 'recharts';
// style={{display: "flex", alingItems:"center", justifyContent:"center"}}

const data = [ 
  {
    "Fecha": 1640995200000,
    "Any": 2022,
    "Mes": 1,
    "Precipitacion": 12.3,
    "Temperatura": 10.2,
    "SPI": -0.7837726661,
    "SPI Historico": -0.8165071595
    },
    {
    "Fecha": 1643673600000,
    "Any": 2022,
    "Mes": 2,
    "Precipitacion": 1.9,
    "Temperatura": 11.8,
    "SPI": -1.0022194416,
    "SPI Historico": -1.0465032184
    },
    {
    "Fecha": 1646092800000,
    "Any": 2022,
    "Mes": 3,
    "Precipitacion": 89.5,
    "Temperatura": 10.8,
    "SPI": 0.837774552,
    "SPI Historico": 0.890771277
    },
    {
    "Fecha": 1648771200000,
    "Any": 2022,
    "Mes": 4,
    "Precipitacion": 38.9,
    "Temperatura": 14.1,
    "SPI": -0.2250530288,
    "SPI Historico": -0.2282480091
    },
    {
    "Fecha": 1651363200000,
    "Any": 2022,
    "Mes": 5,
    "Precipitacion": 20,
    "Temperatura": 20.7,
    "SPI": -0.6220380342,
    "SPI Historico": -0.646221616
    },
    {
    "Fecha": 1654041600000,
    "Any": 2022,
    "Mes": 6,
    "Precipitacion": 9.5,
    "Temperatura": 24.7,
    "SPI": -0.8425852595,
    "SPI Historico": -0.8784291754
    },
    {
    "Fecha": 1656633600000,
    "Any": 2022,
    "Mes": 7,
    "Precipitacion": 2.7,
    "Temperatura": 26.7,
    "SPI": -0.9854158435,
    "SPI Historico": -1.0288112138
    },
    {
    "Fecha": 1659312000000,
    "Any": 2022,
    "Mes": 8,
    "Precipitacion": 67.9,
    "Temperatura": 27.2,
    "SPI": 0.3840774029,
    "SPI Historico": 0.4130871548
    },
    {
    "Fecha": 1661990400000,
    "Any": 2022,
    "Mes": 9,
    "Precipitacion": 13.1,
    "Temperatura": 22.5,
    "SPI": -0.766969068,
    "SPI Historico": -0.798815155
    },
    {
    "Fecha": 1664582400000,
    "Any": 2022,
    "Mes": 10,
    "Precipitacion": 11.3,
    "Temperatura": 20.7,
    "SPI": -0.8047771637,
    "SPI Historico": -0.8386221652
    },
    {
    "Fecha": 1667260800000,
    "Any": 2022,
    "Mes": 11,
    "Precipitacion": 7.6,
    "Temperatura": 15.2,
    "SPI": -0.882493805,
    "SPI Historico": -0.9204476861
    },
    {
    "Fecha": 1669852800000,
    "Any": 2022,
    "Mes": 12,
    "Precipitacion": 33,
    "Temperatura": 12.6,
    "SPI": -0.3489795649,
    "SPI Historico": -0.3587265425
    }];
  
const monthNames = [
      'January', 'February', 'March', 'April', 'May', 'June',
      'July', 'August', 'September', 'October', 'November', 'December'
    ];

const Charts = () => {
  
  return ( 
    <div style={{ height: '500px' }}>
      
    <ResponsiveContainer width="100%" height="100%">
    <ComposedChart
    width={800}
    height={500}
    data={data}
    margin={{
      top: 20,
      right: 20,
      bottom: 20,
      left: 20
    }}
  >
    <CartesianGrid stroke="#f5f5f5" />
    <XAxis dataKey="Mes" tickFormatter={(monthIndex) => monthNames[monthIndex - 1]}/>
    

    <YAxis dataKey="SPI" domain={['dataMin', 100]} />

    <Tooltip />
    <Legend />
    <Area type="monotone" dataKey="SPI" fill="#8884d8" stroke="#8884d8" />
    <Bar dataKey="Precipitacion" barSize={20} fill="#413ea0" />
    <Line type="monotone" dataKey="Temperatura" stroke="red"/>

    {/* <Scatter dataKey="" fill="red" /> */}

    <ReferenceLine y={0} stroke="#000" />
  </ComposedChart>
  </ResponsiveContainer>
  </div>
  )
}

export default Charts

  