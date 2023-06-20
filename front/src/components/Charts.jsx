import React from 'react';
import useFetch from "../hooks/useFetch"
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

  
const monthNames = [
      'January', 'February', 'March', 'April', 'May', 'June',
      'July', 'August', 'September', 'October', 'November', 'December'
    ];

const Charts = () => {
  
  const {data} = useFetch('https://tech4good-backend-production.up.railway.app/api/data')
  // const fetchData = async()
  
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
    <XAxis dataKey="Any" tickFormatter={(monthIndex) => monthNames[monthIndex - 1]}/>
    

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

  