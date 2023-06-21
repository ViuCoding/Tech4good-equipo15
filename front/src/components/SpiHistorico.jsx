import React from 'react';
import { ResponsiveContainer, AreaChart, CartesianGrid, XAxis, YAxis, Tooltip, Area } from 'recharts';
import data from '../../../data/SPI.json';

const SpiHistorico = () => {
  const spiHistoricoDataObj = data["SPI Historico"] || {};
  const spiHistoricoData = Object.entries(spiHistoricoDataObj).map(([timestamp, value]) => ({
    timestamp: Number(timestamp),
    value: value,
  }));

  const slicedData = spiHistoricoData.filter((entry) => {
    const year = new Date(entry.timestamp).getFullYear();
    return year >= 2023 && year <= 2024;
  });

  const formatXAxisTick = (timestamp) => {
    return new Date(timestamp).getFullYear().toString();
  };

  return (
    <div className="h-96">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          width={500}
          height={400}
          data={slicedData}
          margin={{
            top: 10,
            right: 30,
            left: 0,
            bottom: 0,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" tickFormatter={formatXAxisTick} />
          <YAxis />
          <Tooltip />
          <Area type="monotone" dataKey="value" stroke="#1C315E" fill="#227C70" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default SpiHistorico;
 