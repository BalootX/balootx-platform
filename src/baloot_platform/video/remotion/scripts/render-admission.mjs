export const validateViewerBuyerAdmission = (bundle) => {
  const seats = ["north", "west", "south", "east"];
  const buyers = (bundle.scenes ?? []).flatMap(scene => scene.layers ?? [])
    .filter(layer => layer.id === "public-lifecycle").map(layer => layer.content?.buyer);
  const viewer = bundle.observerSeat;
  if (!seats.includes(viewer) || !buyers.length || !seats.includes(buyers[0])
      || buyers.some(buyer => buyer !== buyers[0])) {
    throw new Error("viewer/buyer render admission requires measured valid, consistent seats");
  }
  if (viewer === buyers[0]) {
    throw new Error("viewer=buyer renders are barred pending approved layout repair");
  }
  return {viewer, buyer: buyers[0]};
};
