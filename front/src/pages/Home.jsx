import styled from "styled-components";
import TextContent from "../components/TextContent";
import Charts from "../components/charts";
export const StyledContainer = styled.div`
  width: 80%;
  max-width: 1200px;
  margin: 0 auto;
`;

export default function Home() {
  return (
    <StyledContainer>
      <TextContent />
  
      <Charts />
    </StyledContainer>
  );
}

// #1C315E primario
//#227C70 secundario
//#88A47C compl
//#E6E2C3 compl2
