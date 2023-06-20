import styled from "styled-components";
import PropTypes from "prop-types";

const CardContainer = styled.div`
  max-width: 280px;
  padding-bottom: 2rem;
  margin: 1rem auto;
  transition: all 0.5s ease;
  opacity: 80%;
  box-shadow: 0 5px 5px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);
  border-radius: 12px;

  &:hover {
    opacity: 100%;
    transform: rotateZ(2deg);
  }
`;

const CardThumbnail = styled.div`
  border-radius: 12px 12px 0 0;
  min-height: 180px;
  transition: all 0.5s ease;
`;

const CardDetails = styled.div`
  min-height: 230px;
  border-radius: 0 0 12px 12px;
`;

const SubHeader = styled.h3`
  color: #227c70;
  font-size: 20px;
  font-weight: 600;
  padding: 0.8rem 0.8rem;
`;

const CardText = styled.p`
  padding: 0 0.8rem;
  color: #1c315e;
`;

Card.propTypes = {
  cardImg: PropTypes.string.isRequired,
  cardHeader: PropTypes.string.isRequired,
  cardText: PropTypes.string.isRequired,
};

export default function Card({ cardImg, cardHeader, cardText }) {
  return (
    <CardContainer>
      <CardThumbnail
        style={{
          background: `url(${cardImg}) center/70% no-repeat`,
        }}></CardThumbnail>
      <CardDetails>
        <SubHeader>{cardHeader}</SubHeader>
        <CardText>{cardText}</CardText>
      </CardDetails>
    </CardContainer>
  );
}
