import styled from "styled-components";

import TextContent from "../components/TextContent";
import Charts from "../components/charts";
import SpiGrafico from "../components/SpiGrafico ";
import Card from "../components/Card";

import InfoIcon from "../assets/info.png";
import Alert from "../assets/alert.png";
import IdeaIcon from "../assets/idea.png";

export const StyledContainer = styled.div`
  width: 80%;
  max-width: 1200px;
  margin: 0 auto;
`;

const CardFlex = styled.div`
  padding: 2rem 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-direction: column;
  gap: 1.5rem;

  @media (min-width: 768px) {
    flex-direction: row;
  }
`;

export default function Home() {
  return (
    <StyledContainer>
      <TextContent />
      <CardFlex>
        <Card
          cardImg={InfoIcon}
          cardHeader='Consumo de agua'
          cardText='En Barcelona, en 2021, el consumo de agua potable de red fue de 88,04 hm³, 15,73 hm³ menos que en 2007, año de la última sequía, lo que representa una reducción del 15,16%.'
        />
        <Card
          cardImg={Alert}
          cardHeader='Alerta'
          cardText='El año pasado fue el sexto más seco de la historia de España y el más caluroso desde 1961. Las lluvias acumuladas fueron un 16% menores que el promedio y la temperatura media diaria superó los 15°C por primera vez.'
        />
        <Card
          cardImg={IdeaIcon}
          cardHeader='Posibles soluciones'
          cardText='Cuatro posibles soluciones con las que mitigar las consecuencias de la sequía: desalación de agua, reutilización de agua regenerada, recarga de acuíferos, digitalización del agua.'
        />
      </CardFlex>

      <Charts />
      <SpiGrafico />
    </StyledContainer>
  );
}

// #1C315E primario
//#227C70 secundario
//#88A47C compl
//#E6E2C3 compl2
