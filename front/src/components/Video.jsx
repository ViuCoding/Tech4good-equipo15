import styled from "styled-components";

const StyledVideoContainer = styled.div`
  padding: 2rem 0;
  display: flex;
  justify-content: center;
`;

export default function Video() {
  return (
    <StyledVideoContainer>
      <iframe
        width='560'
        height='315'
        src='https://www.youtube.com/embed/vZyppA94AJg?controls=0&amp;start=11'
        title='YouTube video player'
        allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share'></iframe>
    </StyledVideoContainer>
  );
}
