import Logo from "../assets/logo.png";
import GitHub from "../assets/github.png";
import Insta from "../assets/instagram.png";
import Linkedin from "../assets/linkedin.png";
import RSS from "../assets/rss.png";

import styled from "styled-components";
import { StyledContainer } from "../pages/Home";

const StyledFooter = styled.footer`
  padding: 2rem 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-direction: column;
  gap: 1rem;

  @media (min-width: 768px) {
    flex-direction: row;
  }
`;

const LogoStyled = styled.img`
  width: 50px;
`;

const IconStyled = styled.img`
  width: 30px;
`;

const IconsContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 20px;
`;

const StyledLink = styled.a`
  color: #1c315e;
  font-weight: 600;
  font-size: 0.9rem;
`;

export default function Footer() {
  return (
    <StyledContainer>
      <StyledFooter>
        <div>
          <LogoStyled src={Logo} />
        </div>

        <IconsContainer>
          <StyledLink href='#'>Home</StyledLink>
          <StyledLink href='#'>Precipitaciones</StyledLink>
          <StyledLink href='#'>Depositos de agua</StyledLink>
          <StyledLink href='#'>Media de temperatura</StyledLink>
        </IconsContainer>

        <IconsContainer>
          <IconStyled src={GitHub} />
          <IconStyled src={Insta} />
          <IconStyled src={Linkedin} />
          <IconStyled src={RSS} />
        </IconsContainer>
      </StyledFooter>
    </StyledContainer>
  );
}
