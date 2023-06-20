import styled from "styled-components";

const StyledH1 = styled.h1`
  padding: 1rem 0;
  font-size: 32px;
  font-weight: 800;
  color: #227c70;
`;

const StyledP = styled.p`
  color: #1c315e;

  @media (min-width: 768px) {
    columns: 2;
  }
`;

export default function TextContent() {
  return (
    <>
      <StyledH1>La sequía en Barcelona</StyledH1>
      <StyledP>
        Barcelona es una ciudad que regularmente sufre situaciones de sequía. La
        última gran sequía tuvo lugar en 2007. Ante este problema, el
        Ayuntamiento, en colaboración con otras administraciones, ha establecido
        un Protocolo de actuación por riesgo de sequía de la ciudad de
        Barcelona, que tiene el objetivo de prevenir y avanzarse a posibles
        sequías y, en caso de producirse, minimizar las consecuencias,
        estableciendo medidas preventivas y definiendo un modelo de actuación de
        los servicios municipales.
      </StyledP>
    </>
  );
}
